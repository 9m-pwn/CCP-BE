import tensorflow as tf
import cv2
import numpy as np

from scipy.ndimage import zoom

def make_gradcam_heatmap(model, img_array, target_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[model.input],
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return heatmap.numpy()
    heatmap /= max_val
    return heatmap.numpy()


def overlay_heatmap_on_image(original_img, heatmap, alpha=0.4):
    """original_img shape (H,W,3) float32 [0–1]"""
    h, w = heatmap.shape
    heatmap = zoom(heatmap, (original_img.shape[0]/h, original_img.shape[1]/w))
    hm_uint8 = np.uint8(255 * heatmap)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    over = cv2.addWeighted((original_img*255).astype(np.uint8), 1-alpha, hm_color, alpha, 0)
    return cv2.cvtColor(over, cv2.COLOR_BGR2RGB)