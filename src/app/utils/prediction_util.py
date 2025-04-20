import numpy as np
from tensorflow.keras.models import load_model

CLASS_MAP = {
    0: "Normal",
    1: "LSIL (CIN1)",
    2: "HSIL (CIN2/3)",
    3: "Invasive Carcinoma"
}

def load_trained_model(model_path):
    return load_model(model_path)

def predict_label_and_confidence(model, input_tensor):
    preds = model.predict(input_tensor)
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))
    return idx, conf
