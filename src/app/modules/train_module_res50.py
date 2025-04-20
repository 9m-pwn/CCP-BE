import os
import logging
from pathlib import Path
import pandas as pd
import tensorflow as tf
import json
import time
import redis
import numpy as np
import cv2
from PIL import Image, ImageFile
from scipy.ndimage import zoom

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
    GlobalMaxPooling2D,
    Reshape,
    Dense,
    Activation,
    Add,
    Multiply,
    Concatenate,
    Conv2D,
    Lambda,
    Dropout
)
import tensorflow.keras.backend as K

# Enable loading of truncated JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Redis connection URL
REDIS_URL = "redis://localhost:6379"

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
tf.get_logger().setLevel('ERROR')

# In-memory training status
training_status = {"current_epoch": 0, "total_epochs": 0, "in_progress": False, "status": "not_started", "metrics": {}}

# Utility to verify image integrity

def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        logging.warning(f"Corrupt image skipped: {path} -> {e}")
        return False

# Push training status to Redis

def update_training_status(status: dict):
    client = redis.from_url(REDIS_URL)
    client.set("training_status", json.dumps(status))
    client.close()

class TrainingStatusCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, status_file_path=None):
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

# Grad-CAM utilities
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, eps=1e-8):

    # Build grad model
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
        heatmap = make_gradcam_heatmap(
            self.sample_image, 
            self.model, 
            self.last_conv_layer
        )

        img = (self.sample_image[0] * 255).astype(np.uint8)
        heatmap_resized = zoom(
            heatmap, 
            (
                img.shape[0]/heatmap.shape[0], 
                img.shape[1]/heatmap.shape[1]
            )
        )
        color_map = cv2.applyColorMap(
            (heatmap_resized*255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(img, 0.6, color_map, 0.4, 0)
        fname = os.path.join(self.output_dir, f"gradcam_epoch{epoch+1}.png")
        cv2.imwrite(fname, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        logging.info(f"[GradCAM] saved: {fname}")

# CBAM block definition
def cbam_block(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_dense_one = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_dense_two = Dense(channel, kernel_initializer='he_normal', use_bias=True)

    # Channel attention
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    channel_refined = Multiply()([input_feature, cbam_feature])

    # Spatial attention
    avg_pool_sp = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined)
    max_pool_sp = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined)
    concat = Concatenate(axis=3)([avg_pool_sp, max_pool_sp])
    spatial_map = Conv2D(1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    refined_feature = Multiply()([channel_refined, spatial_map])
    return refined_feature

# Main training function
def train_model(epochs=50, batch_size=32, learning_rate=1e-4):
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    IMAGES_EXCEL = PROJECT_ROOT / "data" / "train" / "case_image.xlsx"
    META_EXCEL   = PROJECT_ROOT / "data" / "train" / "case_metadata.xlsx"


    # Read metadata
    df_img = pd.read_excel(IMAGES_EXCEL)
    df_meta = pd.read_excel(META_EXCEL, header=1)
    if 'Case Number' not in df_meta.columns:
        df_meta.rename(columns={'Unnamed: 0':'Case Number'}, inplace=True)
    df = pd.merge(df_img, df_meta, on='Case Number', how='left')
    df['path'] = df.apply(lambda r: os.path.join(PROJECT_ROOT, 'data/train', f"Case {int(r['Case Number']):03d}", str(r['File']).strip()), axis=1)
    df = df[df['path'].apply(lambda p: os.path.exists(p) and check_image(p))]

    # Label mapping
    def map_label(x):
        s = str(x).lower()
        if 'invasive' in s or 'adenocarcinoma' in s: return 3
        if any(k in s for k in ['hsil','cin2','cin3']): return 2
        if any(k in s for k in ['lsil','cin1']): return 1
        return 0
    df['label'] = df['Histopathology'].apply(map_label)
    num_classes = df['label'].nunique()

    # Dataset pipeline
    paths, labels = df['path'].tolist(), df['label'].tolist()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _decode_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224,224))
        img = preprocess_input(img)
        return img, label

    ds = ds.map(_decode_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    buffer_size = min(len(paths), 1024)
    ds = ds.shuffle(buffer_size, seed=42)
    val_size = int(0.15 * len(paths))
    test_size = val_size
    ds_val   = ds.take(val_size)
    ds_test  = ds.skip(val_size).take(test_size)
    ds_train = ds.skip(val_size + test_size)
    ds_train = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_val   = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Model building
    base = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
    base.trainable = False
    x = cbam_block(base.output, ratio=8)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Grad-CAM callback setup
    sample_path = df['path'].iloc[0]
    sample_img = cv2.imread(sample_path)
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    sample_img = cv2.resize(sample_img, (224,224)).astype(np.float32) / 255.0
    sample_input = np.expand_dims(sample_img, axis=0)
    gradcam_cb = SimpleGradCAM(
        sample_image=sample_input,
        last_conv_layer='conv5_block3_out',
        output_dir=os.path.join(PROJECT_ROOT, 'logs/heatmaps'),
        interval=1
    )

    # Other callbacks
    status_cb = TrainingStatusCallback(epochs, status_file_path=os.path.join(PROJECT_ROOT, 'logs/train_status.json'))
    ckpt_path = os.path.join(PROJECT_ROOT, 'app', 'model_from_resnet50_cbam.keras')
    cbs = [
        status_cb,
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1),
        TensorBoard(log_dir=os.path.join(PROJECT_ROOT, 'logs/fit', time.strftime("%Y%m%d-%H%M%S")), histogram_freq=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        gradcam_cb
    ]

    # Train model
    history = model.fit(ds_train, epochs=epochs, validation_data=ds_val, callbacks=cbs)

    # Save history and final model
    with open(os.path.join(PROJECT_ROOT,'logs/history.json'),'w') as f:
        json.dump(history.history, f)
    model.save(ckpt_path)
    return history.history


if __name__ == '__main__':
    train_model()
