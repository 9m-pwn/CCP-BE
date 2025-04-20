import cv2
import numpy as np

def load_image(path, target_size=(224,224)):
    """โหลดรูปจากไฟล์ → BGR→RGB → resize → float32 /255"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size).astype(np.float32) / 255.0
    return img

def preprocess_for_resnet50(img):
    """รับ image float32 [0–1] → ResNet50 preprocess_input"""
    from tensorflow.keras.applications.resnet50 import preprocess_input
    # preprocess_input ของ ResNet50 คาด [0–255]
    img = (img * 255.0).astype(np.float32)
    return preprocess_input(img)

def make_input_tensor(path, preprocess_fn, target_size=(224,224)):
    """จบครบ: โหลด+preprocess → ขยายมิติเป็น (1,H,W,3)"""
    img = load_image(path, target_size)
    if preprocess_fn:
        img = preprocess_fn(img)
    return np.expand_dims(img, axis=0)
