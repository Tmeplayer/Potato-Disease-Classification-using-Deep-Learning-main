import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from streamlit_extras.add_vertical_space import add_vertical_space

leaf_model = tf.keras.models.load_model("leaf_checker_model.h5")  
disease_model = tf.keras.models.load_model("model.h5") 

disease_classes = ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']

st.set_page_config(page_title="Potato Diagnosis", layout="centered")
st.markdown("<h1 style='text-align: center;'>🌿 Potato Leaf Disease Classifier</h1>", unsafe_allow_html=True)
add_vertical_space(2)

def is_potato_leaf(img):
    img_resized = img.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = leaf_model.predict(img_array)[0][0]
    return prediction > 0.4  

def predict_disease(img):
    img_resized = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = disease_model.predict(img_array)
    predicted_class = disease_classes[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)
    return predicted_class, confidence

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    add_vertical_space(1)


    if is_potato_leaf(image):
        predicted_class, confidence = predict_disease(image)
        st.markdown(f"<h4 style='color: green;'>🟢 Potato Leaf Detected</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: orange;'>🔍 Predicted Disease: {predicted_class}<br>Confidence: {confidence}%</h4>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h4 style='color: red;'>🚫 This image is not recognized as a potato leaf. Please upload a valid leaf image.</h4>", unsafe_allow_html=True)
