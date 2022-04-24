import numpy as np
from tensorflow import keras
import cv2

class_names = {
    1: "Tumor Positive",
    2: "Tumor Negative"
}

def pre_proc_img(image, size=224, scale=0.2):
    """
    Pre Process Image
    """
    center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
    width_scaled, height_scaled = image.shape[1] * scale, image.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    image = image[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    image = cv2.resize(image, (size, size))
    return image

def predict(image):
    """
    Tumor Prediction
    """
    model_path = "model/best_model.h5"
    model = keras.models.load_model(model_path)
    image = pre_proc_img(image=image)
    image = np.expand_dims(image, axis=-1)
    input = np.expand_dims(image, axis=0)
    out = model.predict(input)
    pred = np.argmax(out, axis=-1)
    pred_class = class_names[pred]
    return pred_class, out[pred]
