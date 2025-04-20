import os
import logging
import pandas as pd
import tensorflow as tf
import json
import time
# import asyncio
# import redis.asyncio as redis
import redis
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from scipy.ndimage import zoom

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    ReduceLROnPlateau
)
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
)
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

# Redis URL
REDIS_URL = "redis://localhost:6379"

# Enable loading of truncated JPEGs
tf.get_logger().setLevel('ERROR')
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Utility to verify images

def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        logging.warning(f"Corrupt image skipped: {path} -> {e}")
        return False

# Callback to push training status to Redis
tf.get_logger().setLevel('INFO')
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def update_training_status(status: dict):
    client = redis.from_url(REDIS_URL)
    client.set("training_status", json.dumps(status))
    client.close()

class TrainingStatusCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, status_file_path=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.status_file_path = status_file_path

    # Write status to file and Redis 
    def _write_status(self):
        if self.status_file_path:
            os.makedirs(os.path.dirname(self.status_file_path), exist_ok=True)
            with open(self.status_file_path, "w") as f:
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

# Simple Grad-CAM callback 
# FYI: Grad-CAM is a technique to visualize where a model is looking when making predictions.
# It uses the gradients of the last convolutional layer to create a heatmap that highlights important regions in the image.
# This is a simplified version of Grad-CAM, and it may not be as robust as the original implementation.
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
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
    return heatmap.numpy()

class SimpleGradCAM(tf.keras.callbacks.Callback):
    def __init__(self, sample_image, last_conv_layer, output_dir="heatmaps", interval=1):
        super().__init__()
        self.sample_image = sample_image
        self.last_conv_layer = last_conv_layer
        self.output_dir = output_dir
        self.interval = interval
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval != 0:
            return
        heatmap = make_gradcam_heatmap(
            self.sample_image, self.model, self.last_conv_layer
        )
        # overlay
        img = self.sample_image[0]
        heatmap_resized = zoom(heatmap, (224/heatmap.shape[0], 224/heatmap.shape[1]))
        overlay = cv2.addWeighted((img*255).astype(np.uint8), 0.6,
                                  cv2.applyColorMap((heatmap_resized*255).astype(np.uint8), cv2.COLORMAP_JET),
                                  0.4, 0)
        fname = os.path.join(self.output_dir, f"gradcam_epoch{epoch+1}.png")
        cv2.imwrite(fname, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        logging.info(f"[GradCAM] saved: {fname}")

import tensorflow.keras.backend as K
def cbam_block(input_feature, ratio=8):
    """ Convolutional Block Attention Module (CBAM) """
    channel = input_feature.shape[-1]
    shared_dense_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True)
    shared_dense_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True)
    
    # 1) Channel attention
    #  a) average pooling
    avg_pool = GlobalAveragePooling2D()(input_feature)                     # → (batch, C)
    avg_pool = Reshape((1,1,channel))(avg_pool)                            # → (batch, 1,1,C)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

     #  b) max pooling
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    #  c) combine & sigmoid
    cbam = Add()([avg_pool, max_pool])
    cbam = Activation('sigmoid')(cbam)                                     # → channel attention map
    channel_refined = Multiply()([input_feature, cbam])                    # apply

     # 2) Spatial attention
    #  a) along channel axis→ avg & max pools
    avg_pool_spatial = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined)
    max_pool_spatial = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined)
    concat = Concatenate(axis=3)([avg_pool_spatial, max_pool_spatial])      # → (batch,H,W,2)

    #  b) conv + sigmoid
    spatial_map = Conv2D(1, kernel_size=7, strides=1, padding='same',
                         activation='sigmoid',
                         kernel_initializer='he_normal',
                         use_bias=False)(concat)

    refined_feature = Multiply()([channel_refined, spatial_map])
    return refined_feature

# In-memory status structure
training_status = {"current_epoch":0, "total_epochs":0, "in_progress":False, "status":"not_started", "metrics":{}}

# Main training function
def train_model(epochs=50, batch_size=32, learning_rate=1e-4):
    # paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    images_excel = os.path.join(project_root, "data/processed/train/case_image.xlsx")
    meta_excel   = os.path.join(project_root, "data/processed/train/case_metadata.xlsx")

    # load metadata
    df_images = pd.read_excel(images_excel)
    df_meta   = pd.read_excel(meta_excel, header=1)
    if "Case Number" not in df_meta.columns:
        df_meta.rename(columns={"Unnamed: 0":"Case Number"}, inplace=True)
    df = pd.merge(df_images, df_meta, on="Case Number", how="left")
    df["path"] = df.apply(lambda r: os.path.join(project_root, f"data/processed/train/Case {int(r['Case Number']):03d}", str(r['File']).strip()), axis=1)
    df = df[df['path'].apply(lambda p: os.path.exists(p) and check_image(p))]
    
    # mapping labels
    def map_hist(x):
        s=str(x).lower()
        if 'invasive' in s: return 3
        if any(k in s for k in ['hsil','cin2','cin3']): return 2
        if any(k in s for k in ['lsil','cin1']): return 1
        return 0
    df['label']=df['Histopathology'].apply(map_hist)
    num_classes=df['label'].nunique()

    # 1. Build dataset
    paths, labels = df['path'].tolist(), df['label'].tolist()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _decode_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224,224))
        img = preprocess_input(img)  # ResNet50 preprocessing
        return img, label

    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE)

    # 2. Shuffle with a capped buffer size
    buffer_size = min(len(paths), 1024)
    ds = ds.shuffle(buffer_size, seed=42)

    # 3. Split 70/15/15
    num = len(paths)
    n_val  = int(0.15 * num)
    n_test = int(0.15 * num)

    ds_val   = ds.take(n_val)
    ds_train = ds.skip(n_val + n_test)

    # 4. Batch and prefetch
    ds_train = ds_train.batch(batch_size).prefetch(AUTOTUNE)
    ds_val   = ds_val.batch(batch_size).prefetch(AUTOTUNE)

    # 5. Build & compile model
    base = ResNet50(
        include_top=False, 
        weights='imagenet', 
        input_shape=(224,224,3),
    )
    base.trainable = False
    
    
    x = base.output # output of ResNet50
    x = cbam_block(x, ratio=8) # apply CBAM block to ResNet50 output

    # x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = GlobalAveragePooling2D()(base.output) # global pooling to reduce dimensions
    x = Dense(256, activation='relu')(x) # fully connected layer
    x = Dropout(0.3)(x) # dropout for regularization to prevent overfitting

    outputs = Dense(num_classes, activation='softmax')(x)
    

    model = Model(inputs=base.input, outputs=outputs)

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    model.summary()
    
    # เลเยอร์ conv สุดท้ายของ ResNet50
    last_conv = 'conv5_block3_out'

    # Prepare Grad-CAM
    sample_path = df['path'].iloc[0]
    img = cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224)).astype(np.float32)
    sample_input = np.expand_dims(preprocess_input(img), axis=0)
    
    gradcam_cb = SimpleGradCAM(
        sample_input, 
        last_conv_layer=last_conv, 
        output_dir=os.path.join(project_root,'logs/heatmaps'), 
        interval=1
    )

    # callbacks
    ckpt = os.path.join(project_root, 'app/model_from_resnet50.keras')
    cb = [
        TrainingStatusCallback(epochs, status_file_path=os.path.join(project_root,'logs/train_status.json')),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, verbose=1),
        TensorBoard(log_dir=os.path.join(project_root,'logs/fit',time.strftime("%Y%m%d-%H%M%S")), histogram_freq=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        gradcam_cb
    ]

    # train
    history = model.fit(ds_train, epochs=epochs, validation_data=ds_val, callbacks=cb)
    # save history & final model
    with open(os.path.join(project_root,'logs/history.json'),'w') as f:
        json.dump(history.history, f)
    model.save(ckpt)
    return history.history

if __name__ == '__main__':
    train_model()