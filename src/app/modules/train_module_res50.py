import os
import logging
import json
import time
from pathlib import Path

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

# enable faulthandler and allow truncated JPEGs
import faulthandler; faulthandler.enable()
ImageFile.LOAD_TRUNCATED_IMAGES = True

# setup logging
default_fmt = "%(levelname)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=default_fmt)
logger = logging.getLogger(__name__)

REDIS_URL = "redis://localhost:6379"
training_status = {"current_epoch":0, "total_epochs":0, "in_progress":False, "status":"not_started", "metrics":{}}


def check_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.warning(f"Skipped corrupt image: {path} -> {e}")
        return False


def update_training_status(status: dict):
    client = redis.from_url(REDIS_URL)
    client.set("training_status", json.dumps(status))
    client.close()


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


def train_model(epochs=50, batch_size=32, learning_rate=1e-4):
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

    # ตรวจสอบ/สร้างคอลัมน์ Label
    if 'Label' not in df.columns:
        df.rename(columns={'Histopathology': 'Label'}, inplace=True)
    # แมปข้อความ → หมายเลข
    unique_labels = sorted(df['Label'].unique())
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    df['label_id'] = df['Label'].map(label2id)

    all_paths  = df['path'].tolist()
    all_labels = df['label_id'].tolist()

    valid_paths, valid_labels = [], []
    for p, l in zip(all_paths, all_labels):
        if not os.path.exists(p):
            logger.warning(f"Missing file, skip: {p}")
            continue
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception:
            logger.warning(f"Corrupt image, skip: {p}")
            continue
        valid_paths.append(p)
        valid_labels.append(l)

    logger.info(f"Using {len(valid_paths)} samples")

    AUTOTUNE = tf.data.AUTOTUNE
    IMG_SIZE = (224, 224)

    def load_and_preprocess(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3, try_recover_truncated=True)
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_input(img)
        return img, label

    ds = (
        tf.data.Dataset
        .from_tensor_slices((valid_paths, valid_labels))
        .repeat()
        .shuffle(buffer_size=len(valid_paths), seed=42)
        .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
        .apply(tf.data.experimental.ignore_errors())
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    base = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
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
    cbs = [
        TrainingStatusCallback(epochs, status_file_path=str(ROOT/'logs'/'train_status.json')),
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(str(ckpt), monitor='loss', save_best_only=True, verbose=1),
        TensorBoard(log_dir=str(ROOT/'logs'/'fit'/time.strftime("%Y%m%d-%H%M%S")), histogram_freq=1),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    steps = len(valid_paths) // batch_size
    history = model.fit(
        ds,
        epochs=epochs,
        steps_per_epoch=steps,
        callbacks=cbs
    )

    with open(ROOT/'logs'/'history.json', 'w') as f:
        json.dump(history.history, f)
    model.save(str(ckpt))
    return history.history


if __name__ == '__main__':
    train_model()
