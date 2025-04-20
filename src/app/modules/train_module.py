import os
import logging
import pandas as pd
import tensorflow as tf
import json
import time
import asyncio
import redis.asyncio as redis
import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()  # ตรวจสอบไฟล์ว่าเสียหายหรือไม่
        return True
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Global variable to store training status (used as an in-memory cache)
training_status = {
    "current_epoch": 0,
    "total_epochs": 0,
    "in_progress": False,
    "status": "not_started",  # other possible values: "in_progress", "completed"
    "metrics": {}
}

REDIS_URL = "redis://localhost:6379"

# ฟังก์ชันสำหรับอัปเดตสถานะการเทรนลงใน Redis
async def update_training_status_in_redis(status: dict):
    redis_client = redis.from_url(REDIS_URL)
    await redis_client.set("training_status", json.dumps(status))
    await redis_client.close()

class TrainingStatusCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, status_file_path=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.status_file_path = status_file_path

    async def _update_status(self):
        if self.status_file_path:
            os.makedirs(os.path.dirname(self.status_file_path), exist_ok=True)
            with open(self.status_file_path, "w") as f:
                json.dump(training_status, f)
        await update_training_status_in_redis(training_status)

    def on_train_begin(self, logs=None):
        training_status["total_epochs"] = self.total_epochs
        training_status["current_epoch"] = 0
        training_status["in_progress"] = True
        training_status["status"] = "in_progress"
        training_status["metrics"] = {}
        print('on train begin:', training_status)
        asyncio.run(self._update_status())

    def on_epoch_end(self, epoch, logs=None):
        training_status["current_epoch"] = epoch + 1
        training_status["metrics"] = {
            "loss": logs.get("loss"),
            "accuracy": logs.get("accuracy"),
            "val_loss": logs.get("val_loss"),
            "val_accuracy": logs.get("val_accuracy")
        }
        print('on epoch end:', training_status)
        asyncio.run(self._update_status())
        print(f"[Callback] Updated training_status: {training_status}")
        print(f"Epoch {epoch+1}/{self.total_epochs} - loss: {logs.get('loss')}, accuracy: {logs.get('accuracy')}")

    def on_train_end(self, logs=None):
        training_status["in_progress"] = False
        training_status["status"] = "completed"
        print('on train end:', training_status)
        asyncio.run(self._update_status())

def train_model(epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001) -> dict:
    global training_status

    # 1. Set up project root and read data
    logging.info("Setting up project root and reading data...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logging.info(f"Project root: {project_root}")

    images_excel_path = os.path.join(project_root, "data", "processed", "train", "case_image.xlsx")
    metadata_excel_path = os.path.join(project_root, "data", "processed", "train", "case_metadata.xlsx")
    logging.info(f"Reading images from: {images_excel_path}")
    logging.info(f"Reading metadata from: {metadata_excel_path}")

    df_images = pd.read_excel(images_excel_path)
    df_meta = pd.read_excel(metadata_excel_path, header=1)
    if "Case Number" not in df_meta.columns:
        df_meta.rename(columns={"Unnamed: 0": "Case Number"}, inplace=True)

    # 2. Merge DataFrames
    logging.info("Merging dataframes...")
    df_merged = pd.merge(df_images, df_meta, on="Case Number", how="left")

    # 3. Create image paths
    logging.info("Creating image paths...")
    def get_image_path(row):
        case_num = int(row["Case Number"])
        case_folder = f"Case {case_num:03d}"
        file_name = str(row["File"]).strip()
        return os.path.join(project_root, "data", "processed", "train", case_folder, file_name)
    df_merged["image_path"] = df_merged.apply(get_image_path, axis=1).astype(str)

    # 4. Check if image files exist
    logging.info("Checking if image files exist...")
    df_merged["exists"] = df_merged["image_path"].apply(lambda x: os.path.exists(x) and check_image(x))
    missing_files = df_merged[~df_merged["exists"]]
    if not missing_files.empty:
        logging.warning("Missing or corrupt files:")
        logging.warning(missing_files[["Case Number", "File", "image_path"]])
    df_merged = df_merged[df_merged["exists"]].copy()

    # 5. Map labels from Histopathology column
    logging.info("Mapping labels from Histopathology column...")
    def map_histopathology(label_str):
        if pd.isna(label_str):
            return 0
        label_str = str(label_str).lower().strip()
        # Mapping: 0 = Normal, 1 = LSIL (CIN1), 2 = HSIL (CIN2/3), 3 = Invasive Carcinoma
        if "invasive" in label_str or "adenocarcinoma" in label_str:
            return 3
        elif "hsil" in label_str or "cin2" in label_str or "cin3" in label_str:
            return 2
        elif "lsil" in label_str or "cin1" in label_str:
            return 1
        else:
            return 0
    df_merged["label"] = df_merged["Histopathology"].apply(map_histopathology)
    num_classes = max(df_merged["label"].nunique(), 1)
    logging.info(f"Number of classes (from Histopathology): {num_classes}")

    # 6. Create TensorFlow Dataset
    logging.info("Creating TensorFlow Dataset...")
    image_paths = df_merged["image_path"].tolist()
    labels = df_merged["label"].tolist()
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.cache()

    def process_image(image_path, label, target_size=(224, 224)):
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, target_size)
            image = image / 255.0
            return image, label
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            return tf.zeros(target_size + (3,), dtype=tf.float32), label

    dataset = dataset.map(lambda path, label: process_image(path, label),
                          num_parallel_calls=tf.data.AUTOTUNE)

    dataset_size = len(image_paths)
    val_size = int(dataset_size * 0.1)
    train_size = dataset_size - val_size

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    # 7. Data Augmentation: เพิ่มความหลากหลายให้กับข้อมูลเทรน
    logging.info("Adding data augmentation...")
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.1)
        return image, label
    
    train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 8. Create or update model using Transfer Learning
    model_path = os.path.join(project_root, "app", "model_from_IARC.keras")
    
    # Remove model if it exists to create a new one
    if os.path.exists(model_path):
        logging.info(f"Removing existing model at {model_path}...")
        os.remove(model_path)

    logging.info("Creating a new model with Transfer Learning...")
    
    
    model = tf.keras.models.Sequential([

        tf.keras.layers.InputLayer((224,224,3)),
        # Conv Block 1
        tf.keras.layers.Conv2D(32,3,padding='same',activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        # Conv Block 2
        tf.keras.layers.Conv2D(64,3,padding='same',activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        # Conv Block 3
        tf.keras.layers.Conv2D(128,3,padding='same',activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        # Conv Block 4 (ขยาย feature map)
        tf.keras.layers.Conv2D(256,3,padding='same',activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        # Global pooling แทน Flatten เพื่อลด overfitting
        tf.keras.layers.GlobalAveragePooling2D(),

        # Dense head
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        # 5 คลาส (stage 0–IV)
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # 9. Set up callbacks: EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, and TrainingStatusCallback
    logging.info("Setting up callbacks...")
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    checkpoint_path = os.path.join(project_root, "app", "model_from_IARC.keras")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    log_dir = os.path.join(project_root, "logs", "fit", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    status_file_path = os.path.join(project_root, "logs", "training_status.json")
    os.makedirs(os.path.dirname(status_file_path), exist_ok=True)
    status_callback = TrainingStatusCallback(total_epochs=epochs, status_file_path=status_file_path)

    callbacks = [
        status_callback, 
        early_stopping, 
        model_checkpoint, 
        tensorboard_callback, 
        reduce_lr, 
    ]

    # 10. Train the model (Initial Training + Fine-Tuning)
    logging.info("Starting training...")
    history_initial = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )

    # Save training history for frontend visualization
    history_path = os.path.join(project_root, "logs", "training_history.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history_initial.history, f)
    logging.info(f"Training history saved to {history_path}")

    # 11. Save the final model
    logging.info("Saving the final model...")
    model.save(model_path)
    logging.info(f"Training complete, model saved at {model_path}")

    return history_initial.history

if __name__ == "__main__":
    train_model()