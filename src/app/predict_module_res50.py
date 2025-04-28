# src/app/modules/predict_module_res50.py

import os
import numpy as np
import tensorflow as tf
import cv2
from scipy.ndimage import zoom
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

# ถ้ามี Lambda layer ฝังอยู่ ให้ enable unsafe deserialization
tf.keras.config.enable_unsafe_deserialization()

# ชื่อเลเยอร์สุดท้ายสำหรับ Grad-CAM และแมปผลลัพธ์เป็นชื่อคลาส
LAST_CONV_LAYER = "conv5_block3_out"
LABELS = ["Normal", "LSIL", "HSIL", "Invasive"]

# Cache โมเดลเพื่อโหลดครั้งเดียว
_model_cache: dict[str, tf.keras.Model] = {}

def get_model(model_path: str) -> tf.keras.Model:
    """
    โหลดหรือคืนค่าจาก cache ถ้าโหลดไปแล้ว
    """
    if model_path not in _model_cache:
        _model_cache[model_path] = load_model(model_path, compile=False)
    return _model_cache[model_path]

def preprocess_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    อ่านภาพจากดิสก์ (full-res RGB uint8),
    แล้ว resize+preprocess สำหรับโมเดล (224×224 float32, subtract mean)
    คืนค่า (orig_rgb_uint8, preprocessed_tensor)
    """
    # --- อ่าน original full resolution ---
    img_bgr_full = cv2.imread(image_path)
    if img_bgr_full is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)

    # --- ทำ resize+preprocess สำหรับโมเดล ---
    img_rgb = img_rgb_full.astype(np.float32)
    img_resized = cv2.resize(img_rgb, (224, 224))
    inp = preprocess_input(img_resized)       # [-mean … +mean]
    inp = np.expand_dims(inp, axis=0)         # (1,224,224,3)

    return img_rgb_full, inp

def predict_image(image_path: str, model_path: str) -> tuple[str, float]:
    """
    ทำ inference และคืน (label, confidence%)
    """
    orig, inp = preprocess_image(image_path)
    model = get_model(model_path)
    preds = model.predict(inp, verbose=0)
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0])) * 100.0
    label = LABELS[idx] if idx < len(LABELS) else str(idx)
    return label, conf

def make_gradcam_heatmap(model: tf.keras.Model,
                         img_tensor: np.ndarray,
                         last_conv_layer: str = LAST_CONV_LAYER,
                         eps: float = 1e-8) -> np.ndarray:
    """
    สร้าง heatmap shape (h,w) ค่าระหว่าง 0–1
    img_tensor: (1,H,W,3) หลัง preprocess_input
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv = conv_outputs[0]  # (h,w,channels)
    heatmap = conv @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
    return heatmap.numpy()

def overlay_heatmap_on_image(original_img: np.ndarray,
                             heatmap: np.ndarray,
                             alpha: float = 0.4) -> np.ndarray:
    """
    ผสม heatmap ลงบนภาพ RGB [0–255] (ทั้ง hot และ cool zone)
    คืนค่าเป็นภาพ RGB [0–255]
    """
    # 1) upscale heatmap ให้เท่าขนาด original
    h, w = heatmap.shape
    zoom_factors = (original_img.shape[0] / h, original_img.shape[1] / w)
    heatmap_resized = zoom(heatmap, zoom_factors)

    # 2) สร้างสีจาก heatmap
    heat_uint8 = np.uint8(255 * heatmap_resized)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    # 3) blend — แต่ cv2.applyColorMap คืน BGR เราต้องแปลง original → BGR ก่อน
    orig_bgr    = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(orig_bgr, 1 - alpha,
                                  heat_color, alpha, 0)

    # 4) คืนเป็น RGB
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


def overlay_hot_only(original_img: np.ndarray,
                     heatmap: np.ndarray,
                     alpha: float = 0.4,
                     cool_threshold: float = 0.2) -> np.ndarray:
    """
    ผสม heatmap ลงบนภาพ RGB [0–255] แต่ให้ cool zone (heatmap < cool_threshold) เป็น transparent
    คืนค่าเป็นภาพ RGB [0–255]
    """
    # 1) ขยาย heatmap ให้เท่าขนาด original
    h, w = heatmap.shape
    zoom_factors = (original_img.shape[0] / h, original_img.shape[1] / w)
    heatmap_resized = zoom(heatmap, zoom_factors)
    
    # 2) สร้างสี heatmap
    heat_uint8 = np.uint8(255 * heatmap_resized)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)  # BGR
    
    # 3) สร้าง mask สำหรับ hot zone
    #    True ที่ค่า heatmap_resized >= cool_threshold
    mask = heatmap_resized >= cool_threshold
    mask = np.stack([mask]*3, axis=-1)  # shape (H,W,3)
    
    # 4) blend เฉพาะจุด hot zone
    orig_bgr = original_img.astype(np.float32)
    heat_bgr = heat_color.astype(np.float32)
    
    # เริ่มจาก copy ภาพเดิมไว้
    blended = orig_bgr.copy()
    
    # ที่ mask==True ให้ทำการ addWeighted
    blended[mask] = orig_bgr[mask] * (1 - alpha) + heat_bgr[mask] * alpha
    
    # 5) คืนเป็น RGB uint8
    blended = blended.astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)