import os
import logging
import json
import time
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import redis
import tensorflow as tf
import cv2
from PIL import Image, ImageFile
from scipy.ndimage import zoom

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    ReduceLROnPlateau
)
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Reshape,
    Dense,
    Dropout,
    Add,
    Activation,
    Multiply,
    GlobalMaxPooling2D,
    GlobalAveragePooling2D,
    Concatenate,
    Conv2D
)
import tensorflow.keras.backend as K
from app.settings import get_redis_url

# enable faulthandler and allow truncated JPEGs
import faulthandler; faulthandler.enable()
ImageFile.LOAD_TRUNCATED_IMAGES = True

# setup logging
default_fmt = "%(levelname)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=default_fmt)
logger = logging.getLogger(__name__)

REDIS_URL = get_redis_url()
training_status = {"current_epoch":0, "total_epochs":0, "in_progress":False, "status":"not_started", "metrics":{}}
EXCLUDED_LABELS = {"not done"}
# Dev guide:
# - Keep RANDOM_SEED fixed for reproducible experiments.
# - Change only if you intentionally start a new baseline and will compare again from scratch.
RANDOM_SEED = 42
# Dev guide:
# - Split is at Case Number level (not image level) to prevent data leakage.
# - Edit these ratios when dataset size changes significantly.
# - Revisit if validation/test counts become too small to represent real production cases.
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# Stage-2 class weights can be manually boosted for clinically important classes.
CLASS_WEIGHT_MULTIPLIER = {
    "Invasive": 1.5,
    "Normal": 1.2,
}


def check_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.warning(f"Skipped corrupt image: {path} -> {e}")
        return False


def update_training_status(status: dict):
    try:
        client = redis.from_url(REDIS_URL)
        client.set("training_status", json.dumps(status))
        client.close()
    except Exception as exc:
        logger.warning(f"Redis update skipped: {exc}")


class TrainingStatusCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs: int, status_file_path: str=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.status_file_path = status_file_path

    def _write_status(self):
        if self.status_file_path:
            os.makedirs(os.path.dirname(self.status_file_path), exist_ok=True)
            with open(self.status_file_path, 'w') as f:
                json.dump(training_status, f)
        update_training_status(training_status)

    def on_train_begin(self, logs=None):
        training_status.update({
            "total_epochs": self.total_epochs,
            "current_epoch": 0,
            "in_progress": True,
            "status": "in_progress",
            "metrics": {}
        })
        self._write_status()

    def on_epoch_end(self, epoch, logs=None):
        training_status["current_epoch"] = epoch + 1
        training_status["metrics"] = {
            "loss": logs.get("loss"),
            "accuracy": logs.get("accuracy"),
            "val_loss": logs.get("val_loss"),
            "val_accuracy": logs.get("val_accuracy")
        }
        self._write_status()

    def on_train_end(self, logs=None):
        training_status.update({"in_progress": False, "status": "completed"})
        self._write_status()


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, eps=1e-8):
    grad_model = tf.keras.models.Model(
        inputs=[model.input],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
    return heatmap.numpy()


class SimpleGradCAM(tf.keras.callbacks.Callback):
    def __init__(self, sample_image, last_conv_layer, output_dir="heatmaps/train", interval=1):
        super().__init__()
        self.sample_image = sample_image
        self.last_conv_layer = last_conv_layer
        self.output_dir = output_dir
        self.interval = interval
        os.makedirs(self.output_dir, exist_ok=True)

    def set_model(self, model):
        super().set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval != 0:
            return
        heatmap = make_gradcam_heatmap(self.sample_image, self.model, self.last_conv_layer)
        img = (self.sample_image[0] * 255).astype(np.uint8)
        heat_resz = zoom(heatmap, (img.shape[0]/heatmap.shape[0], img.shape[1]/heatmap.shape[1]))
        cmap = cv2.applyColorMap((heat_resz*255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, cmap, 0.4, 0)
        out_path = os.path.join(self.output_dir, f"gradcam_epoch{epoch+1}.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        logger.info(f"[GradCAM] saved: {out_path}")


def normalize_histopathology_label(raw_label: str) -> Optional[str]:
    """
    Normalize free-text labels to stable classes used by training/inference.
    Returns one of: Normal, LSIL, HSIL, Invasive; otherwise None.
    """
    text = str(raw_label).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None

    # Keep invasive rule first: strings can contain both invasive and HSIL terms.
    invasive_tokens = (
        "invasive",
        "microinvasive",
        "carcinoma",
        "adenocarcinoma",
        "squamous cell cancer",
        "cancer",
    )
    if any(tok in text for tok in invasive_tokens):
        return "Invasive"

    hsil_tokens = (
        "hsil",
        "cin2",
        "cin 2",
        "cin3",
        "cin 3",
        "vain 3",
        "vin 3",
        "high-grade",
        "high grade",
    )
    if any(tok in text for tok in hsil_tokens):
        return "HSIL"

    lsil_tokens = (
        "lsil",
        "cin1",
        "cin 1",
        "hpv changes",
        "low-grade",
        "low grade",
    )
    if any(tok in text for tok in lsil_tokens):
        return "LSIL"

    if "normal" in text or "negative for intraepithelial lesion" in text:
        return "Normal"

    return None


def compute_distribution_by_label(rows, class_names):
    """
    rows: iterable[(path, label_id, case_number)]
    class_names: ordered class labels matching label_id
    """
    result = []
    for class_id, label in enumerate(class_names):
        class_rows = [r for r in rows if int(r[1]) == class_id]
        unique_cases = {int(r[2]) for r in class_rows}
        result.append(
            {
                "class_id": class_id,
                "label": label,
                "sample_count": int(len(class_rows)),
                "unique_case_count": int(len(unique_cases)),
            }
        )

    total_samples = sum(x["sample_count"] for x in result)
    for row in result:
        row["sample_ratio"] = float(row["sample_count"] / total_samples) if total_samples > 0 else 0.0
    return result


def build_class_weight(train_rows, class_names):
    """
    Build class weights from train split only.
    Weight formula: total / (num_classes * class_count), with optional multipliers.
    """
    counts = {class_id: 0 for class_id in range(len(class_names))}
    for _, label_id, _ in train_rows:
        counts[int(label_id)] += 1

    total = float(sum(counts.values()))
    num_classes = float(max(len(class_names), 1))
    class_weight = {}
    for class_id, count in counts.items():
        base = (total / (num_classes * float(count))) if count > 0 else 1.0
        label = class_names[class_id]
        multiplier = float(CLASS_WEIGHT_MULTIPLIER.get(label, 1.0))
        class_weight[class_id] = float(base * multiplier)
    return class_weight


def cbam_block(input_feature, ratio=8):
    channel = int(input_feature.shape[-1])
    dense1 = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    dense2 = Dense(channel, kernel_initializer='he_normal', use_bias=True)

    avg = GlobalAveragePooling2D()(input_feature)
    avg = Reshape((1,1,channel))(avg)
    avg = dense2(dense1(avg))

    mx = GlobalMaxPooling2D()(input_feature)
    mx = Reshape((1,1,channel))(mx)
    mx = dense2(dense1(mx))

    ca = Activation('sigmoid')(Add()([avg, mx]))
    x = Multiply()([input_feature, ca])

    avg_sp = GlobalAveragePooling2D(keepdims=True)(x)
    mx_sp  = GlobalMaxPooling2D(keepdims=True)(x)

    concat = Concatenate(axis=-1)([avg_sp, mx_sp])
    sa = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', use_bias=False)(concat)
    return Multiply()([x, sa])


def split_case_numbers(case_numbers, seed=RANDOM_SEED):
    # Split by case id (not by image) to avoid data leakage across subsets.
    unique_cases = np.array(sorted({int(c) for c in case_numbers}))
    if unique_cases.size < 3:
        raise ValueError("Need at least 3 unique Case Number values for train/val/test split")

    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if total_ratio <= 0:
        raise ValueError("Split ratios must be positive")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_cases)

    train_count = int(round(shuffled.size * (TRAIN_RATIO / total_ratio)))
    val_count = int(round(shuffled.size * (VAL_RATIO / total_ratio)))

    train_count = max(1, train_count)
    val_count = max(1, val_count)
    if train_count + val_count >= shuffled.size:
        val_count = max(1, shuffled.size - train_count - 1)
        train_count = max(1, shuffled.size - val_count - 1)

    train_cases = set(shuffled[:train_count].tolist())
    val_cases = set(shuffled[train_count:train_count + val_count].tolist())
    test_cases = set(shuffled[train_count + val_count:].tolist())

    if not test_cases:
        moved = val_cases.pop()
        test_cases.add(moved)

    return train_cases, val_cases, test_cases


def compute_classification_metrics(y_true, y_pred, class_names):
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    eps = 1e-12

    precision = tp / (predicted + eps)
    recall = tp / (support + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    total = float(cm.sum())
    accuracy = float(tp.sum() / total) if total > 0 else 0.0
    macro_precision = float(np.mean(precision)) if num_classes > 0 else 0.0
    macro_recall = float(np.mean(recall)) if num_classes > 0 else 0.0
    macro_f1 = float(np.mean(f1)) if num_classes > 0 else 0.0
    weighted_precision = float(np.sum(precision * support) / (support.sum() + eps))
    weighted_recall = float(np.sum(recall * support) / (support.sum() + eps))
    weighted_f1 = float(np.sum(f1 * support) / (support.sum() + eps))

    per_class = []
    for idx, class_name in enumerate(class_names):
        per_class.append(
            {
                "class_id": idx,
                "label": class_name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
                "false_positive": int(predicted[idx] - tp[idx]),
                "false_negative": int(support[idx] - tp[idx]),
            }
        )

    return {
        "accuracy": accuracy,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        },
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def train_model(epochs=50, batch_size=32, learning_rate=1e-4):
    # Load image-level and metadata-level sheets, then merge by Case Number.
    # Dev guide:
    # - Update these file paths when data source format/location changes.
    # - Keep column names stable ("Case Number", "File", "Label"/"Histopathology").
    # - If upstream schema changes, fix mapping here first before tuning model.
    ROOT = Path(__file__).resolve().parent.parent.parent
    IMG_EXCEL = ROOT / "data" / "train" / "case_image.xlsx"
    META_EXCEL = ROOT / "data" / "train" / "case_metadata.xlsx"

    df_img = pd.read_excel(IMG_EXCEL)
    df_meta = pd.read_excel(META_EXCEL, header=1)
    if 'Case Number' not in df_meta.columns:
        df_meta.rename(columns={'Unnamed: 0': 'Case Number'}, inplace=True)

    df = df_img.merge(df_meta, on='Case Number', how='left')
    df['path'] = df.apply(
        lambda r: str(ROOT / "data" / "train" / f"Case {int(r['Case Number']):03d}" / r['File']),
        axis=1
    )

    # standardize label column name and filter out excluded labels
    # Dev guide:
    # - Add values to EXCLUDED_LABELS when business confirms labels are not trainable.
    # - If many rows are excluded, check with domain owner before training.
    if 'Label' not in df.columns:
        df.rename(columns={'Histopathology': 'Label'}, inplace=True)

    df['Label'] = df['Label'].astype(str).str.strip()
    df = df[~df['Label'].str.lower().isin(EXCLUDED_LABELS)].copy()
    if df.empty:
        raise ValueError("No training rows left after excluding labels: not done")

    # Normalize noisy free-text labels into stable classes.
    df["raw_label"] = df["Label"]
    df["Label"] = df["raw_label"].apply(normalize_histopathology_label)

    unknown_label_rows = df[df["Label"].isna()].copy()
    if not unknown_label_rows.empty:
        logger.warning(
            "Dropped %s rows with unmapped labels. Example raw labels: %s",
            len(unknown_label_rows),
            unknown_label_rows["raw_label"].dropna().astype(str).head(5).tolist(),
        )
    df = df[df["Label"].notna()].copy()
    if df.empty:
        raise ValueError("No rows left after label normalization. Check label mapping rules.")

    unique_labels = sorted(df['Label'].dropna().unique())
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    df['label_id'] = df['Label'].map(label2id)

    # Validate files early so train/val/test subsets contain only readable images.
    # Dev guide:
    # - Frequent "Missing file" logs usually indicate data pipeline/path issues, not model issues.
    # - Fix data integrity first; do not compensate by only increasing epochs.
    valid_rows = []
    for path, label_id, case_number in df[["path", "label_id", "Case Number"]].itertuples(index=False, name=None):
        label_id = int(label_id)
        case_number = int(case_number)
        if not os.path.exists(path):
            logger.warning(f"Missing file, skip: {path}")
            continue
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            logger.warning(f"Corrupt image, skip: {path}")
            continue
        valid_rows.append((path, label_id, case_number))

    logger.info(f"Using {len(valid_rows)} samples")
    if not valid_rows:
        raise ValueError("No valid training samples found under src/data/train")

    train_cases, val_cases, test_cases = split_case_numbers([r[2] for r in valid_rows])

    train_rows = [r for r in valid_rows if r[2] in train_cases]
    val_rows = [r for r in valid_rows if r[2] in val_cases]
    test_rows = [r for r in valid_rows if r[2] in test_cases]

    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Split produced an empty subset. Please add more data.")

    train_paths, train_labels = zip(*[(r[0], r[1]) for r in train_rows])
    val_paths, val_labels = zip(*[(r[0], r[1]) for r in val_rows])
    test_paths, test_labels = zip(*[(r[0], r[1]) for r in test_rows])

    logger.info(
        "Split by Case Number -> train: %s samples (%s cases), val: %s samples (%s cases), test: %s samples (%s cases)",
        len(train_paths),
        len(train_cases),
        len(val_paths),
        len(val_cases),
        len(test_paths),
        len(test_cases),
    )

    distribution_report = {
        "classes": unique_labels,
        "num_classes": int(len(unique_labels)),
        "dropped_unmapped_label_rows": int(len(unknown_label_rows)),
        "unmapped_label_examples": unknown_label_rows["raw_label"].dropna().astype(str).unique().tolist()[:20],
        "overall": compute_distribution_by_label(valid_rows, unique_labels),
        "train": compute_distribution_by_label(train_rows, unique_labels),
        "val": compute_distribution_by_label(val_rows, unique_labels),
        "test": compute_distribution_by_label(test_rows, unique_labels),
    }
    class_weight = build_class_weight(train_rows, unique_labels)
    distribution_report["class_weight"] = {
        unique_labels[int(class_id)]: float(weight) for class_id, weight in class_weight.items()
    }

    AUTOTUNE = tf.data.AUTOTUNE
    IMG_SIZE = (224, 224)
    # Dev guide:
    # - Keep IMG_SIZE=(224,224) unless you intentionally change model backbone/input.
    # - If GPU memory is insufficient, reduce batch_size first before changing image size.

    def load_and_preprocess(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3, try_recover_truncated=True)
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_input(img)
        return img, label

    # Build independent datasets for training, validation, and final holdout test.
    train_ds = (
        tf.data.Dataset
        .from_tensor_slices((list(train_paths), list(train_labels)))
        .repeat()
        .shuffle(buffer_size=len(train_paths), seed=RANDOM_SEED)
        .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
        .ignore_errors()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset
        .from_tensor_slices((list(val_paths), list(val_labels)))
        .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
        .ignore_errors()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset
        .from_tensor_slices((list(test_paths), list(test_labels)))
        .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
        .ignore_errors()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test_eval_ds = (
        tf.data.Dataset
        .from_tensor_slices((list(test_paths), list(test_labels)))
        .map(
            lambda path, label: (
                load_and_preprocess(path, label)[0],
                label,
                path,
            ),
            num_parallel_calls=AUTOTUNE
        )
        .ignore_errors()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    base = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    # Dev guide:
    # - Phase 1 default: freeze backbone for stable transfer learning.
    # - Consider unfreezing top backbone layers only after val metrics plateau for multiple runs.
    # - When unfreezing, reduce learning_rate (for example x0.1) to avoid catastrophic forgetting.
    base.trainable = False
    x = cbam_block(base.output, ratio=8)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(len(unique_labels), activation='softmax')(x)
    model = Model(base.input, out)

    model.compile(
        optimizer=Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    ckpt = ROOT / 'app' / 'model_from_resnet50_cbam.keras'
    labels_path = ckpt.with_suffix('.labels.json')
    # Dev guide:
    # - EarlyStopping/RLROP are the first knobs to adjust when overfit or unstable training appears.
    # - Increase patience only if val_loss still trends down but needs more epochs.
    # - Do not remove ModelCheckpoint; prediction service depends on the saved best model file.
    cbs = [
        TrainingStatusCallback(epochs, status_file_path=str(ROOT/'logs'/'train_status.json')),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(str(ckpt), monitor='val_loss', save_best_only=True, verbose=1),
        TensorBoard(log_dir=str(ROOT/'logs'/'fit'/time.strftime("%Y%m%d-%H%M%S")), histogram_freq=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    steps = max(1, len(train_paths) // batch_size)
    val_steps = max(1, len(val_paths) // batch_size)
    # Train with validation monitoring; test set is evaluated once after fitting.
    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=cbs,
        class_weight=class_weight,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    history.history["test_loss"] = [float(test_loss)]
    history.history["test_accuracy"] = [float(test_accuracy)]

    y_true = []
    y_pred = []
    misclassified = []
    for imgs, labels, paths in test_eval_ds:
        probs = model(imgs, training=False).numpy()
        pred_ids = np.argmax(probs, axis=1)
        true_ids = labels.numpy().astype(np.int64)
        pred_conf = probs[np.arange(probs.shape[0]), pred_ids]

        y_true.extend(true_ids.tolist())
        y_pred.extend(pred_ids.tolist())

        path_values = paths.numpy()
        for i in range(len(pred_ids)):
            if int(pred_ids[i]) == int(true_ids[i]):
                continue
            path_value = path_values[i]
            if isinstance(path_value, bytes):
                path_text = path_value.decode("utf-8", errors="replace")
            else:
                path_text = str(path_value)
            misclassified.append(
                {
                    "path": path_text,
                    "true_id": int(true_ids[i]),
                    "true_label": unique_labels[int(true_ids[i])],
                    "pred_id": int(pred_ids[i]),
                    "pred_label": unique_labels[int(pred_ids[i])],
                    "confidence": float(pred_conf[i]),
                }
            )

    report = compute_classification_metrics(
        np.array(y_true, dtype=np.int64),
        np.array(y_pred, dtype=np.int64),
        unique_labels,
    )
    misclassified.sort(key=lambda x: x["confidence"], reverse=True)
    report["top_false_positive_classes"] = sorted(
        (
            {"label": row["label"], "class_id": row["class_id"], "count": row["false_positive"]}
            for row in report["per_class"]
        ),
        key=lambda x: x["count"],
        reverse=True,
    )[:5]
    report["top_false_negative_classes"] = sorted(
        (
            {"label": row["label"], "class_id": row["class_id"], "count": row["false_negative"]}
            for row in report["per_class"]
        ),
        key=lambda x: x["count"],
        reverse=True,
    )[:5]
    report["num_test_samples_scored"] = int(len(y_true))
    report["num_misclassified"] = int(len(misclassified))
    history.history["test_macro_f1"] = [float(report["macro_avg"]["f1"])]
    history.history["test_weighted_f1"] = [float(report["weighted_avg"]["f1"])]

    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with open(ROOT/'logs'/'history.json', 'w') as f:
        json.dump(history.history, f)
    with open(ROOT/'logs'/'classification_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(ROOT/'logs'/'top_misclassified.json', 'w', encoding='utf-8') as f:
        json.dump(misclassified[:50], f, ensure_ascii=False, indent=2)
    with open(ROOT/'logs'/'class_distribution.json', 'w', encoding='utf-8') as f:
        json.dump(distribution_report, f, ensure_ascii=False, indent=2)
    # Dev guide:
    # - First file to inspect after each run: logs/classification_report.json.
    # - Retrain trigger:
    #   1) macro F1 drops from previous accepted baseline,
    #   2) false negatives increase on critical classes,
    #   3) new label set or major data distribution change arrives.
    # - Data fix takes priority over hyperparameter tuning when errors cluster in one class.
    model.save(str(ckpt))
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'labels': unique_labels,
                'id2label': {str(i): lab for i, lab in enumerate(unique_labels)},
                'label2id': label2id,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return history.history


if __name__ == '__main__':
    train_model()


