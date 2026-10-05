"""Brain Tumor MRI Classifier - Streamlit app.

Run:  streamlit run app.py
Expects:  models/best_model.h5  and  models/class_names.json  (produced by the notebook).
The scaling/normalisation layer is part of the saved model, so images only need resizing here.
"""
import json
import os

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.h5")
CLASSES_PATH = os.environ.get("CLASSES_PATH", "models/class_names.json")
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No tumor",
    "pituitary": "Pituitary tumor",
}
LOW_CONFIDENCE = 0.60   # below this, tell the user not to trust the prediction

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model_and_classes():
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def preprocess(image: Image.Image, model) -> np.ndarray:
    h, w = model.input_shape[1:3]                                    # e.g. 224 x 224
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    # Same resize + rounding as the training pipeline (tf.image.resize). Using PIL's resize here would
    # anti-alias differently and quietly shift the input distribution.
    arr = tf.round(tf.image.resize(arr, (h, w))).numpy().astype("float32")
    return np.expand_dims(arr, 0)                                    # 0-255; scaling is inside the model


st.title("🧠 Brain Tumor MRI Classifier")
st.caption("Upload a brain MRI image to get the predicted tumor type with confidence scores.")
st.warning("Educational project, **not a medical device**. Never use it for diagnosis or treatment decisions.")

try:
    model, class_names = load_model_and_classes()
except Exception as e:                                               # missing / incompatible model file
    st.error(f"Could not load the model from `{MODEL_PATH}`. Run the notebook first and copy "
             f"`best_model.h5` and `class_names.json` into the `models/` folder.\n\nDetails: {e}")
    st.stop()

with st.sidebar:
    st.header("About")
    st.write("Classes the model knows:")
    for c in class_names:
        st.write(f"- {DISPLAY_NAMES.get(c, c)}")
    st.write(f"Input size: {model.input_shape[1]} x {model.input_shape[2]}")
    st.caption("Trained on a small public dataset (~2.4k images) with no external hospital validation.")

uploaded = st.file_uploader("Upload an MRI image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = Image.open(uploaded)
    except Exception:
        st.error("This file could not be read as an image.")
        st.stop()

    col_img, col_out = st.columns([1, 1])
    col_img.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analysing..."):
        probs = model.predict(preprocess(image, model), verbose=0)[0]

    top = int(np.argmax(probs))
    label = DISPLAY_NAMES.get(class_names[top], class_names[top])
    col_out.metric("Predicted class", label)
    col_out.metric("Confidence", f"{probs[top] * 100:.1f}%")

    if probs[top] < LOW_CONFIDENCE:
        st.warning("Low confidence: the model is unsure about this image. Treat the result as unreliable.")

    st.subheader("Confidence for every class")
    for name, p in sorted(zip(class_names, probs), key=lambda t: -t[1]):
        st.write(f"**{DISPLAY_NAMES.get(name, name)}** - {p * 100:.1f}%")
        st.progress(float(p))
else:
    st.info("Upload an image to start.")
    