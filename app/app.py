'''
    Brain Tumor classification app using streamlit
'''
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pydicom
import streamlit as st
from tensorflow import keras

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
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    """
    Tumor Prediction
    """
    model_path = "model/best_model.h5"
    model = keras.models.load_model(model_path)
    image = pre_proc_img(image=image)
    out = model.predict(image)
    return out

st.title("Brain Tumor Detector")
st.subheader("A CNN based Brian Tumor Diagnois application")

st.markdown("> Disclaimer : I do not claim this application as a highly accurate Brain Tumor Diagnosis Tool. This Application has not been professionally or academically Vetted. This is purely for Educational Purpose to demonstrate the Potential of AI's help in Medicine.")


uploaded_file = st.file_uploader("Choose an Brain FLAIR MRI image", type=("dcm"))
if uploaded_file is not None:

    image = pydicom.read_file(uploaded_file).pixel_array
    plt.axis("off")
    plt.imshow(image, cmap="gray")
    st.pyplot(plt)
    pred = predict(np.array(image))
    pred = np.argmax(pred)
    result = "Tumor: Positive" if pred == 1 else "Tumor: Negative"
    st.markdown("## **Diagnosed Result:**")
    st.markdown(result)

st.markdown("Developed by [Ajay B](https://github.com/ajaybalakrishnan)")
st.markdown("### Check out the [GitHub Repository](https://github.com/ajaybalakrishnan/Brain-Tumor-Classification)")