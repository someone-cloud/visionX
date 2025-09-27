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

# Debug info
st.write("🔍 MODEL_PATH =", MODEL_PATH)
st.write("📍 Absolute path:", os.path.abspath(MODEL_PATH))
st.write("📦 Exists:", os.path.exists(MODEL_PATH))

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
        st.error(f"❌ Failed to load model from `{path}`.\n\n
