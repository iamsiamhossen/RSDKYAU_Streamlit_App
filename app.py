import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras

st.set_page_config(
    page_title="RSDKYAU Roadside Damage Classification",
    page_icon="🛣️",
    layout="wide"
)

MODEL_PATH = "best_DenseNet121_finetuned.keras"
CLASS_NAMES = ["Pot hole", "crack_dataset", "edge_damage", "open_drain_dataset"]
IMG_SIZE = (224, 224)

@st.cache_resource
def load_trained_model():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_trained_model()

st.title("RSDKYAU Roadside Damage Classification")
st.write(
    "This Streamlit demo uses a fine-tuned DenseNet121 model to classify roadside "
    "infrastructure defects into four categories: Pot hole, crack, edge damage, and open drain."
)

st.sidebar.header("Model Information")
st.sidebar.write("Final Model: Fine-tuned DenseNet121")
st.sidebar.write("Test Accuracy: 97.48%")
st.sidebar.write("Weighted F1-score: 97.50%")
st.sidebar.warning("This app performs image-level classification, not bounding-box object detection.")

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image).astype(np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    img_batch = keras.applications.densenet.preprocess_input(img_batch)

    return img_batch

def predict_image(image):
    img_batch = preprocess_image(image)
    probs = model.predict(img_batch, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs

uploaded_file = st.file_uploader(
    "Upload a roadside damage image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    pred_class, confidence, probs = predict_image(image)

    with col2:
        st.subheader("Prediction Result")

        st.success(f"Predicted Class: {pred_class}")
        st.info(f"Confidence: {confidence * 100:.2f}%")

        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probs
        })

        st.subheader("Class Probability")
        st.bar_chart(prob_df.set_index("Class"))

        st.dataframe(prob_df)

else:
    st.info("Please upload an image to classify.")