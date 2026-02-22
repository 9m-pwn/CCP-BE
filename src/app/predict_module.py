import os
import tensorflow as tf
import numpy as np


# Map model output index to readable labels.
label_mapping = {
    0: "Normal",
    1: "LSIL (CIN1)",
    2: "HSIL (CIN2/3)",
    3: "Invasive Carcinoma"
}


def load_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for prediction.
    - Verify the file exists.
    - Use tf.keras.preprocessing.image.load_img to load and resize.
    - Convert to numpy array and normalize pixels to 0-1.
    - Add batch dimension to shape (1, height, width, channels).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Load and resize image.
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.astype("float32") / 255.0
    # Add batch dimension.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(image_path, model_path="model_from_resnet50_cbam.keras"):
    """
    Load a model from model_path and run prediction for image_path.
    Return predicted label and confidence.
    """
    # Load saved model.
    model = tf.keras.models.load_model(model_path)
    # Load and preprocess image.
    image = load_image(image_path)
    # Run prediction and get probabilities.
    prediction = model.predict(image)
    # Select class index with maximum probability.
    predicted_index = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))
    # Convert class index to readable label.
    predicted_label = label_mapping.get(predicted_index, "Unknown")
    return predicted_label, confidence


if __name__ == "__main__":
    # Example: test predict module directly.
    test_image_path = "path/to/your/test_image.jpg"  # Replace with a valid path.
    try:
        label, conf = predict_image(test_image_path)
        print(f"Predicted Label: {label}, Confidence: {conf:.2f}")
    except Exception as e:
        print(f"Error: {e}")
