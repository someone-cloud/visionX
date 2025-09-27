
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load model once and cache
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("visionx_model.h5")

model = load_model()
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Page config
st.set_page_config(
    page_title="VisionX",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
    <style>
    .main {background-color: #111111; color: #EEEEEE;}
    h1 {text-align: center; color: #00C9A7;}
    .stImage {border-radius: 12px;}
    .stButton>button {background-color: #00C9A7; color: black; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>👁️ VisionX - CIFAR-10 Classifier</h1>", unsafe_allow_html=True)
st.write("Upload an image to see AI classification, confidence bars, and fun facts!")

# Fun facts for all classes
facts = {
    "airplane": "✈️ Some planes, like the Gulfstream G800, can fly at Mach 0.95 — by the time you blink, it could have completed a 100-meter race!",
    "automobile": "🚗 The Bugatti Chiron can accelerate from 0-100 km/h in just 2.4 seconds — faster than a cheetah!",
    "bird": "🦜 Peregrine falcons can dive at over 320 km/h — fastest animal on Earth!",
    "cat": "🐱 A cat’s purring can promote healing and reduce stress — tiny feline therapy!",
    "deer": "🦌 A deer can jump up to 3 meters high and 9 meters long — Olympic-level leaping!",
    "dog": "🐶 Dogs can understand up to 250 words and gestures — super-smart and human-friendly!",
    "frog": "🐸 The Goliath frog can grow over 32 cm and weigh 3 kg — giant among frogs!",
    "horse": "🐴 Horses can sleep both lying down and standing up thanks to a special leg mechanism!",
    "ship": "🚢 The largest ship ever built, the Seawise Giant, was 458 meters long — longer than 4 football fields!",
    "truck": "🚚 The BelAZ 75710 truck can carry 450 tons — heavier than 6 blue whales!"
}

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg","png"])

if uploaded_file:
    # Process image
    img = tf.keras.utils.load_img(uploaded_file, target_size=(32,32))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array/255.0, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)*100

    # Layout: two columns
    col1, col2 = st.columns([1,2])

    # Column 1: uploaded image + prediction
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.subheader(f"🔮 Prediction: **{pred_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
        st.info(f"💡 Fun Fact: {facts.get(pred_class,'No fun fact stored yet!')}")

    # Column 2: confidence chart
    with col2:
        st.markdown("### 📊 Confidence Levels")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(class_names, preds[0], color="#00C9A7")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_ylabel("Probability")
        st.pyplot(fig)