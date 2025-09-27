'''# visionx_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# --- SETTINGS ---
IMG_SIZE = 32  # CIFAR-10 image size
MODEL_PATH = "visionx_model.h5"  # must be in repo root

# --- CLASS NAMES & FUN FACTS ---
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

fun_facts = {
    "airplane": "Some planes like the Gulfstream G800 can fly at Mach 0.95.",
    "automobile": "The world's first practical automobile was built in 1885.",
    "bird": "The fastest bird is the peregrine falcon, reaching 389 km/h.",
    "cat": "Cats can rotate their ears 180 degrees independently.",
    "deer": "Deer have excellent night vision due to a reflective layer behind their retina.",
    "dog": "Dogs have about 220 million scent receptors, far more than humans.",
    "frog": "Some frogs can freeze completely and survive the winter.",
    "horse": "Horses sleep both lying down and standing up.",
    "ship": "The largest ship ever built was the Seawise Giant, 458 meters long.",
    "truck": "The fastest production truck is the Ram 1500 TRX, reaching 100 km/h in 4.5 seconds."
}

# --- LOAD MODEL ---
@st.cache_resource
def load_visionx_model(path):
    if not os.path.exists(path):
        st.error(f"Model file {path} not found in repo root!")
        return None
    return load_model(path)

from tensorflow.keras.models import load_model

# Load your trained model safely
model = load_model("visionx_model.h5", compile=False)

# Recompile to use it
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- STREAMLIT INTERFACE ---
st.title("VisionX: CIFAR-10 Classifier with Fun Facts")
st.write("Upload an image, and VisionX will predict its class with confidence levels.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    # Load image
    img = image.load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, IMG_SIZE, IMG_SIZE, 3)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    st.subheader(f"Prediction: {predicted_class} ({confidence*100:.2f}% confidence)")
    
    # Fun fact
    fact = fun_facts.get(predicted_class, "No fun fact available.")
    st.info(f"Fun Fact: {fact}")

    # Confidence bar chart
    fig, ax = plt.subplots()
    ax.barh(class_names, predictions[0])
    ax.set_xlabel("Confidence")
    ax.set_title("Prediction Confidence for all Classes")
    st.pyplot(fig)'''
from tensorflow.keras.models import load_model

from tensorflow.keras.models import load_model

# Load your existing .h5 model
model = load_model("visionx_model.h5", compile=False)

# Save as new Keras native format
model.save("visionx_model.keras")


