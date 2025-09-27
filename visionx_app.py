
# visionx_app.py

# --- IMPORTS ---
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# ✅ Set Streamlit page config early
st.set_page_config(page_title="VisionX", layout="centered")

# --- SETTINGS ---
IMG_SIZE = 32
MODEL_PATH = "visionx_model.keras"  # This should be in your repo root

# --- CLASS NAMES & FUN FACTS ---
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

fun_facts = {
    "airplane": "Some planes like the Gulfstream G800 can fly at Mach 0.95. this means that by the time you blink your eyes, a G800 would have completed a 100 meter race",
    "automobile": "The airbags in your car inflate in 40 miliseconds at speeds of up to 7242km/h",
    "bird": "The fastest bird is the peregrine falcon, reaching 389 km/h.",
    "cat": "A cat's whiskers are as sensitive as human fingertips.",
    "deer": "Deer have excellent night vision due to a reflective layer behind their retina.",
    "dog": "Dogs evolved from the gray wolf. This means that a cute little french bulldog was once a ferocious hunter.",
    "frog": "The most colorful and vibrant frogs are often the most poisonous ones.",
    "horse": "Horses sleep both lying down and standing up.",
    "ship": "Nuclear powered ships can run for up to 25 years before needing to refuel.",
    "truck": "The BelAZ 75710 is the world's largest truck. it has a capacity of 450 metric tonnes and can reach speeds of up to 64km/h."
}


# --- LOAD MODEL ---
@st.cache_resource
def load_visionx_model(path):
    if not os.path.exists(path):
        st.error(f"🚫 Model file not found at: `{path}`")
        return None
    try:
        model = tf.keras.models.load_model(path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model from `{path}`.\n\n**Details:** {e}")
        return None  # <--- Important to return None here to avoid NoneType errors

# Load model
model = load_visionx_model(MODEL_PATH)

# --- STREAMLIT INTERFACE ---
st.title("🧠 VisionX: CIFAR-10 Image Classifier")
st.write("Upload a CIFAR-10-style image (32×32), and VisionX will predict the class.")

uploaded_file = st.file_uploader("📷 Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    try:
        # Load and preprocess image
        img = image.load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 32, 32, 3)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[0][predicted_index]

        # Show prediction
        st.subheader(f"🔍 Prediction: **{predicted_class}**")
        st.write(f"Confidence: `{confidence * 100:.2f}%`")

        # Fun fact
        fact = fun_facts.get(predicted_class, "No fun fact available.")
        st.info(f"💡 Fun Fact: {fact}")

        # Confidence chart
        fig, ax = plt.subplots()
        ax.barh(class_names, predictions[0], color='skyblue')
        ax.set_xlabel("Confidence")
        ax.set_title("Prediction Confidence for All Classes")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❗ An error occurred while processing the image.\n\n**Details:** {e}")

elif uploaded_file and not model:
    st.error("🚫 Model failed to load. Make sure `visionx_model.keras` exists in your project root.")
