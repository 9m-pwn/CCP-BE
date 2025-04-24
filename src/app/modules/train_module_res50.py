# src/app/modules/train_module_res50.py

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
    GlobalAveragePooling2D,
    Reshape,
    Dense,
    Dropout,
    Add,
    Activation,
    Multiply,
    GlobalMaxPooling2D,
    Lambda,
    Concatenate,
    Conv2D
)
import tensorflow.keras.backend as K

# enable Python-level crash reporting
import faulthandler; faulthandler.enable()
# allow PIL to load truncated JPEGs without error
ImageFile.LOAD_TRUNCATED_IMAGES = True

# configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Redis URL for pushing training status
REDIS_URL = "redis://localhost:6379"
# in-memory structure to track training progress
training_status = {
    "current_epoch": 0,
    "total_epochs": 0,
    "in_progress": False,
    "status": "not_started",
    "metrics": {}
}


def check_image(path: str) -> bool:
    """
    Verify that an image file can be opened without corruption.
    Returns True if ok, False if corrupt.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # PIL will throw if the file is not a valid image
        return True
    except Exception as e:
        logger.warning(f"Skipped corrupt image: {path} → {e}")
        return False


def update_training_status(status: dict):
    """
    Push the latest training_status dict to Redis.
    """
    client = redis.from_url(REDIS_URL)
    client.set("training_status", json.dumps(status))
    client.close()


class TrainingStatusCallback(tf.keras.callbacks.Callback):
    """
    Keras callback that updates training_status on start/end of training and each epoch.
    """
    def __init__(self, total_epochs: int, status_file_path: str = None):
        super().__init__()
        self.total_epochs = total_epochs
        self.status_file_path = status_file_path

    def _write_status(self):
        # Optionally write to a JSON file on disk
        if self.status_file_path:
            os.makedirs(os.path.dirname(self.status_file_path), exist_ok=True)
            with open(self.status_file_path, 'w') as f:
                json.dump(training_status, f)
        # Push to Redis
        update_training_status(training_status)

    def on_train_begin(self, logs=None):
        # Marks training as started
        training_status.update({
            "total_epochs": self.total_epochs,
            "current_epoch": 0,
            "in_progress": True,
            "status": "in_progress",
            "metrics": {}
        })
        self._write_status()

    def on_epoch_end(self, epoch, logs=None):
        # Update after each epoch
        training_status["current_epoch"] = epoch + 1
        training_status["metrics"] = {
            "loss": logs.get("loss"),
            "accuracy": logs.get("accuracy"),
            "val_loss": logs.get("val_loss"),
            "val_accuracy": logs.get("val_accuracy")
        }
        self._write_status()

    def on_train_end(self, logs=None):
        # Marks training as finished
        training_status.update({"in_progress": False, "status": "completed"})
        self._write_status()


def cbam_block(input_feature, ratio=8):
    """
    Convolutional Block Attention Module (CBAM).
    Applies channel attention then spatial attention to the input feature map.
    """
    channel = int(input_feature.shape[-1])

    # Shared MLP for channel attention
    shared_dense_1 = Dense(channel // ratio, activation='relu',
                           kernel_initializer='he_normal', use_bias=True)
    shared_dense_2 = Dense(channel, kernel_initializer='he_normal', use_bias=True)

    # 1) Channel attention
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_2(shared_dense_1(avg_pool))

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_2(shared_dense_1(max_pool))

    channel_attention = Activation('sigmoid')(Add()([avg_pool, max_pool]))
    x = Multiply()([input_feature, channel_attention])

    # 2) Spatial attention
    avg_sp = Lambda(lambda t: K.mean(t, axis=3, keepdims=True))(x)
    max_sp = Lambda(lambda t: K.max(t, axis=3, keepdims=True))(x)
    concat = Concatenate(axis=3)([avg_sp, max_sp])
    spatial_attention = Conv2D(1, kernel_size=7, padding='same',
                               activation='sigmoid', use_bias=False)(concat)

    return Multiply()([x, spatial_attention])


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, eps=1e-8):
    """
    Compute Grad-CAM heatmap for a single image tensor.
    """
    grad_model = tf.keras.models.Model(
        inputs=[model.input],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
    return heatmap.numpy()


class SimpleGradCAM(tf.keras.callbacks.Callback):
    """
    Keras callback to save a Grad-CAM overlay image every `interval` epochs.
    """
    def __init__(self, sample_image, last_conv_layer, output_dir="logs/heatmaps/train", interval=1):
        super().__init__()
        self.sample_image = sample_image  # a (1,H,W,3) numpy array
        self.last_conv_layer = last_conv_layer
        self.output_dir = output_dir
        self.interval = interval
        os.makedirs(self.output_dir, exist_ok=True)

    def set_model(self, model):
        super().set_model(model)  # ensures self.model is set

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval != 0:
            return
        
        # compute and save Grad-CAM overlay
        heatmap = make_gradcam_heatmap(self.sample_image, self.model, self.last_conv_layer)
        img = (self.sample_image[0] * 255).astype(np.uint8) # RGB 

        # ขยายขนาดให้ตรงกับภาพต้นฉบับ
        heat_resz = zoom(
            heatmap,
            (img.shape[0] / heatmap.shape[0], img.shape[1] / heatmap.shape[1])
        )
        # แปลงเป็น uint8 [0–255]
        heat_uint8 = (heat_resz * 255).astype(np.uint8)

        # save pure heatmap
        pure_heatmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        pure_fname = os.path.join(self.output_dir, f"heatmap_epoch{epoch+1}.png")
        cv2.imwrite(pure_fname, pure_heatmap)
        logger.info(f"[GradCAM] saved raw heatmap: {pure_fname}")

        # 4) แปลง pure heatmap → RGB เพื่อผสมกับ img_rgb
        heat_rgb   = cv2.cvtColor(pure_heatmap, cv2.COLOR_BGR2RGB)

        # save overlay image
        overlay   = cv2.addWeighted(img, 0.6, heat_rgb, 0.4, 0)
        over_fname = os.path.join(self.output_dir, f"overlay_epoch{epoch+1}.png")
        cv2.imwrite(over_fname, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        logger.info(f"[GradCAM] saved overlay: {over_fname}")


def train_model(epochs=50, batch_size=32, learning_rate=1e-4):
    """
    Main training function.
    - Loads metadata Excel files
    - Filters out corrupt/missing images
    - Builds tf.data pipeline with AUTOTUNE
    - Constructs ResNet50+CBAM, compiles, and fits model
    - Uses callbacks to checkpoint, early-stop, and report status
    """
    ROOT = Path(__file__).resolve().parent.parent.parent
    IMG_EXCEL = ROOT / "data" / "train" / "case_image.xlsx"
    META_EXCEL = ROOT / "data" / "train" / "case_metadata.xlsx"

    # 1) Load metadata
    df_img = pd.read_excel(IMG_EXCEL)
    df_meta = pd.read_excel(META_EXCEL, header=1)
    if 'Case Number' not in df_meta.columns:
        df_meta.rename(columns={'Unnamed: 0': 'Case Number'}, inplace=True)
    df = df_img.merge(df_meta, on='Case Number', how='left')

    # 2) Map text labels to integers
    def map_label(x):
        s = str(x).lower()
        if 'invasive' in s or 'adenocarcinoma' in s:
            return 3
        if any(k in s for k in ['cin2', 'cin3', 'hsil']):
            return 2
        if any(k in s for k in ['cin1', 'lsil']):
            return 1
        return 0
    df['label'] = df['Histopathology'].apply(map_label)

    # 3) Build and filter file paths
    df['path'] = df.apply(
        lambda r: str(ROOT / "data" / "train" / f"Case {int(r['Case Number']):03d}" / r['File']),
        axis=1
    )
    paths = df['path'].tolist()
    labels = df['label'].tolist()
    valid_paths, valid_labels = [], []
    for p, l in zip(paths, labels):
        if not os.path.exists(p) or not check_image(p):
            continue
        valid_paths.append(p)
        valid_labels.append(l)
    
    N = len(valid_paths)
    logger.info(f"Using {N}/{len(paths)} valid samples")

    # 4) Build tf.data pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    IMG_SIZE = (224, 224)

    def load_and_preprocess(path, label):
        # read JPEG, resize, apply ResNet50 preprocessing
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3, try_recover_truncated=True)
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_input(img)
        return img, label

    # 1) สร้าง Dataset ดิบ
    ds_all = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))

    # 2) Shuffle with capped buffer
    buffer_size = min(N, 1024)
    ds_all = ds_all.shuffle(buffer_size, seed=42)

    # 3) Map → preprocess
    ds_all = ds_all.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    # 4) Split into train / val
    val_size   = int(0.15 * N)
    ds_val     = ds_all.take(val_size)
    ds_train   = ds_all.skip(val_size)

    # 5) Batch & prefetch
    ds_train = ds_train.batch(batch_size).prefetch(AUTOTUNE)
    ds_val   = ds_val.batch(batch_size).prefetch(AUTOTUNE)

   # --- Build model as before ---
    base = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
    base.trainable = False # Freeze base model
    x = cbam_block(base.output, ratio=8) # Apply CBAM block
    x = GlobalAveragePooling2D()(x) # Global average pooling
    x = Dense(256, activation='relu')(x) # Fully connected layer
    x = Dropout(0.3)(x) # Dropout for regularization
    out = Dense(len(set(valid_labels)), activation='softmax')(x) # Output layer
    model = Model(base.input, out)

    model.compile(
        optimizer=Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


    # 6) Prepare callbacks
    ckpt_path = ROOT / 'app' / 'model_from_resnet50_cbam.keras'

    sample_path = valid_paths[0] # Use the first valid image for Grad-CAM
    img = cv2.imread(sample_path)                                    # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                       # → RGB
    img = cv2.resize(img, (224,224)).astype(np.float32) / 255.0      # normalize
    sample_input = np.expand_dims(img, axis=0)                       # shape=(1,224,224,3)

    # 2. สร้าง Grad-CAM callback
    last_conv_layer = 'conv5_block3_out'   # ชื่อ layer สุดท้ายก่อน pooling
    heatmap_dir = os.path.join(ROOT, "logs", "heatmaps", "train")
    os.makedirs(heatmap_dir, exist_ok=True)
    heatmap_cb = SimpleGradCAM(
        sample_image=sample_input,
        last_conv_layer=last_conv_layer,
        output_dir=heatmap_dir,
        interval=1       # จะเซฟทุก epoch
    )

    cbs = [
        TrainingStatusCallback(epochs, status_file_path=str(ROOT/'logs'/'train_status.json')),
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(str(ckpt_path), monitor='loss', save_best_only=True, verbose=1),
        TensorBoard(log_dir=str(ROOT/'logs'/'fit'/time.strftime("%Y%m%d-%H%M%S")), histogram_freq=1),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        heatmap_cb
    ]

    # 7) 
    history = model.fit(
        ds_train, 
        epochs=epochs, 
        callbacks=cbs,
        validation_data=ds_val,
    )

    # 8) Save history and model file
    with open(ROOT/'logs'/'history.json','w') as f:
        json.dump(history.history, f)
    model.save(str(ckpt_path))

    return history.history


if __name__ == '__main__':
    train_model()
