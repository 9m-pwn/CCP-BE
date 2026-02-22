import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

tf.keras.config.enable_unsafe_deserialization()

LAST_CONV_LAYER = "conv5_block3_out"
DEFAULT_LABELS = ["Normal", "LSIL", "HSIL", "Invasive"]
DEFAULT_USE_TWO_STAGE = True
DEFAULT_NORMAL_THRESHOLD = 0.50

_model_cache: dict[str, tf.keras.Model] = {}
_labels_cache: dict[str, list[str]] = {}


def _label_map_path_for_model(model_path: str) -> Path:
    return Path(model_path).with_suffix(".labels.json")


def _load_labels_from_json(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        return [str(x) for x in raw]

    if isinstance(raw, dict):
        if isinstance(raw.get("labels"), list):
            return [str(x) for x in raw["labels"]]
        if isinstance(raw.get("id2label"), dict):
            id2label = raw["id2label"]
            ordered = [label for _, label in sorted(id2label.items(), key=lambda kv: int(kv[0]))]
            return [str(x) for x in ordered]

    raise ValueError(f"Unsupported label map format: {path}")


def _infer_labels_from_training_metadata(model_path: str) -> list[str]:
    src_root = Path(model_path).resolve().parents[1]
    meta_excel = src_root / "data" / "train" / "case_metadata.xlsx"
    if not meta_excel.exists():
        return []

    import pandas as pd

    df_meta = pd.read_excel(meta_excel, header=1)
    if "Case Number" not in df_meta.columns and "Unnamed: 0" in df_meta.columns:
        df_meta = df_meta.rename(columns={"Unnamed: 0": "Case Number"})

    label_col = "Label" if "Label" in df_meta.columns else "Histopathology"
    if label_col not in df_meta.columns:
        return []

    labels = (
        df_meta[label_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return sorted(labels.unique().tolist())


def get_labels(model_path: str, expected_num_classes: Optional[int] = None) -> list[str]:
    if model_path not in _labels_cache:
        labels_path = _label_map_path_for_model(model_path)
        if labels_path.exists():
            _labels_cache[model_path] = _load_labels_from_json(labels_path)
        else:
            inferred = _infer_labels_from_training_metadata(model_path)
            if expected_num_classes and len(inferred) == expected_num_classes:
                _labels_cache[model_path] = inferred
            else:
                _labels_cache[model_path] = DEFAULT_LABELS
    return _labels_cache[model_path]


def get_model(model_path: str) -> tf.keras.Model:
    if model_path not in _model_cache:
        _model_cache[model_path] = load_model(model_path, compile=False)
    return _model_cache[model_path]


def preprocess_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    img_bgr_full = cv2.imread(image_path)
    if img_bgr_full is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)

    img_rgb = img_rgb_full.astype(np.float32)
    img_resized = cv2.resize(img_rgb, (224, 224))
    inp = preprocess_input(img_resized)
    inp = np.expand_dims(inp, axis=0)

    return img_rgb_full, inp


def _predict_single_stage(preds: np.ndarray, labels: list[str]) -> tuple[str, float]:
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0])) * 100.0
    label = labels[idx] if idx < len(labels) else str(idx)
    return label, conf


def _predict_two_stage(
    preds: np.ndarray,
    labels: list[str],
    normal_label: str = "Normal",
    normal_threshold: float = DEFAULT_NORMAL_THRESHOLD,
) -> tuple[str, float]:
    probs = preds[0]
    if normal_label not in labels:
        return _predict_single_stage(preds, labels)

    normal_idx = labels.index(normal_label)
    normal_prob = float(probs[normal_idx])
    if normal_prob >= float(normal_threshold):
        return normal_label, normal_prob * 100.0

    abnormal_indices = [i for i, label in enumerate(labels) if label != normal_label]
    if not abnormal_indices:
        return normal_label, normal_prob * 100.0

    abnormal_probs = np.array([probs[i] for i in abnormal_indices], dtype=np.float32)
    best_pos = int(np.argmax(abnormal_probs))
    best_idx = abnormal_indices[best_pos]
    best_label = labels[best_idx]
    return best_label, float(probs[best_idx]) * 100.0


def predict_image(
    image_path: str,
    model_path: str,
    use_two_stage: bool = DEFAULT_USE_TWO_STAGE,
    normal_threshold: float = DEFAULT_NORMAL_THRESHOLD,
) -> tuple[str, float]:
    _, inp = preprocess_image(image_path)
    model = get_model(model_path)
    num_classes = int(model.output_shape[-1]) if isinstance(model.output_shape, tuple) else None
    labels = get_labels(model_path, expected_num_classes=num_classes)

    preds = model.predict(inp, verbose=0)
    if use_two_stage:
        return _predict_two_stage(preds, labels, normal_label="Normal", normal_threshold=normal_threshold)
    return _predict_single_stage(preds, labels)


def make_gradcam_heatmap(
    model: tf.keras.Model,
    img_tensor: np.ndarray,
    last_conv_layer: str = LAST_CONV_LAYER,
    eps: float = 1e-8,
) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv = conv_outputs[0]
    heatmap = conv @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
    return heatmap.numpy()


def _resize_heatmap_to_image(heatmap: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = heatmap.shape
    zoom_factors = (image_shape[0] / h, image_shape[1] / w)
    return zoom(heatmap, zoom_factors)


def overlay_heatmap_on_image(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    heatmap_resized = _resize_heatmap_to_image(heatmap, original_img.shape)

    heat_uint8 = np.uint8(255 * heatmap_resized)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    orig_bgr = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(orig_bgr, 1 - alpha, heat_color, alpha, 0)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


def overlay_hot_only(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    cool_threshold: float = 0.55,
) -> np.ndarray:
    heatmap_resized = _resize_heatmap_to_image(heatmap, original_img.shape)
    adaptive_threshold = max(cool_threshold, float(np.quantile(heatmap_resized, 0.85)))

    heat_uint8 = np.uint8(255 * heatmap_resized)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    mask = heatmap_resized >= adaptive_threshold
    mask_uint8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask = mask_uint8 > 0
    mask = np.stack([mask] * 3, axis=-1)

    orig_bgr = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)
    heat_bgr = heat_color.astype(np.float32)
    blended = orig_bgr.copy()
    blended[mask] = orig_bgr[mask] * (1 - alpha) + heat_bgr[mask] * alpha

    blended = blended.astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def build_gradcam_overlay(
    model_path: str,
    input_tensor: np.ndarray,
    original_img: np.ndarray,
    focus_suspicious_only: bool = True,
    alpha: float = 0.4,
    suspicious_threshold: float = 0.55,
    last_conv_layer: str = LAST_CONV_LAYER,
) -> np.ndarray:
    model = get_model(model_path)
    heatmap = make_gradcam_heatmap(model, input_tensor, last_conv_layer=last_conv_layer)
    if focus_suspicious_only:
        return overlay_hot_only(
            original_img,
            heatmap,
            alpha=alpha,
            cool_threshold=suspicious_threshold,
        )
    return overlay_heatmap_on_image(original_img, heatmap, alpha=alpha)
