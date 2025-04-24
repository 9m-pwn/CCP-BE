import os
import tensorflow as tf
import numpy as np


# กำหนด dictionary mapping เพื่อแปลง index ที่ได้จากโมเดลไปเป็น label ที่เข้าใจง่าย
label_mapping = {
    0: "Normal",
    1: "LSIL (CIN1)",
    2: "HSIL (CIN2/3)",
    3: "Invasive Carcinoma"
}

def load_image(image_path, target_size=(224, 224)):
    """
    โหลดและ preprocess รูปภาพสำหรับ prediction
    - ตรวจสอบว่าไฟล์มีอยู่จริง
    - ใช้ tf.keras.preprocessing.image.load_img ในการโหลดและปรับขนาดรูป
    - แปลงรูปเป็น numpy array และ normalize ค่า pixel ให้อยู่ในช่วง 0-1
    - เพิ่ม batch dimension เพื่อให้มีรูปแบบ (1, height, width, channels)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    
    # โหลดรูปและปรับขนาด
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.astype("float32") / 255.0
    # เพิ่ม dimension สำหรับ batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path, model_path="model_from_resnet50_cbam.keras"):
    """
    โหลดโมเดลจาก model_path แล้วใช้โมเดลทำนายรูปภาพที่ระบุผ่าน image_path
    คืนค่าเป็น predicted label และ confidence (ความมั่นใจ)
    """
    # โหลดโมเดลที่ถูกบันทึกไว้
    model = tf.keras.models.load_model(model_path)
    # โหลดและ preprocess รูปภาพ
    image = load_image(image_path)
    # ทำการ predict รูปภาพ (ผลลัพธ์เป็น probability array)
    prediction = model.predict(image)
    # หา index ของ class ที่มี probability สูงสุด
    predicted_index = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))
    # ใช้ label_mapping แปลง index เป็น label ที่เข้าใจง่าย
    predicted_label = label_mapping.get(predicted_index, "Unknown")
    return predicted_label, confidence

if __name__ == "__main__":
    # ตัวอย่างการทดสอบ module predict.py เมื่อรันไฟล์นี้โดยตรง
    test_image_path = "path/to/your/test_image.jpg"  # เปลี่ยน path ให้ถูกต้อง
    try:
        label, conf = predict_image(test_image_path)
        print(f"Predicted Label: {label}, Confidence: {conf:.2f}")
    except Exception as e:
        print(f"Error: {e}")
