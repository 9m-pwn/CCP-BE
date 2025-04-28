# src/app/utils/prediction_util.py

import numpy as np
import tensorflow as tf

# 1) Print TF-Keras version
print("TF-Keras version:", tf.keras.__version__)

# 2) Allow loading of Lambda layers (unsafe deserialization)
tf.keras.config.enable_unsafe_deserialization()

from tensorflow.keras.models import load_model

CLASS_MAP = {
    0: "Normal",
    1: "LSIL (CIN1)",
    2: "HSIL (CIN2/3)",
    3: "Invasive Carcinoma"
}

def load_trained_model(model_path: str):
    """
    Load a .keras model that may contain Lambda layers.
    Passing safe_mode=False here tells Keras to skip its safety check.
    """
    return load_model(model_path, safe_mode=False)

def predict_label_and_confidence(model, input_tensor):
    """
    Run inference and return (predicted_index, confidence_score).
    """
    preds = model.predict(input_tensor, compile=False)
    idx  = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))
    return idx, conf
