import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
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

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    return keras.models.load_model(MODEL_PATH, compile=False)

model = load_trained_model()

# ------------------------------------------------------------
# Find DenseNet121 base model for Grad-CAM
# ------------------------------------------------------------
@st.cache_resource
def prepare_gradcam_models():
    base_model = None

    for layer in model.layers:
        if isinstance(layer, keras.Model) and "densenet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        nested_models = [layer for layer in model.layers if isinstance(layer, keras.Model)]
        if len(nested_models) > 0:
            base_model = nested_models[0]

    if base_model is None:
        return None, None, None

    last_conv_layer = None
    for layer in reversed(base_model.layers):
        try:
            if len(layer.output.shape) == 4:
                last_conv_layer = layer
                break
        except Exception:
            continue

    if last_conv_layer is None:
        return None, None, None

    base_index = model.layers.index(base_model)
    classifier_layers = model.layers[base_index + 1:]

    feature_extractor = keras.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    classifier_input = keras.Input(shape=base_model.output.shape[1:])
    x = classifier_input

    for layer in classifier_layers:
        try:
            x = layer(x, training=False)
        except Exception:
            x = layer(x)

    classifier_model = keras.Model(classifier_input, x)

    return feature_extractor, classifier_model, last_conv_layer.name

feature_extractor, classifier_model, gradcam_layer_name = prepare_gradcam_models()

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def prepare_raw_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype(np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_array, img_batch

def predict_image(image):
    img_array, img_batch = prepare_raw_image(image)

    # Important:
    # Do NOT apply densenet.preprocess_input here because preprocessing
    # was already included inside the saved training model.
    probs = model.predict(img_batch, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs, img_array

def generate_gradcam(img_array, pred_index=None):
    if feature_extractor is None or classifier_model is None:
        return None

    # For Grad-CAM we bypass the full model and pass image directly to DenseNet base.
    # Therefore preprocessing is needed here.
    input_tensor = np.expand_dims(img_array.copy(), axis=0)
    input_tensor = keras.applications.densenet.preprocess_input(input_tensor)

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = feature_extractor(input_tensor, training=False)
        tape.watch(conv_outputs)

        preds = classifier_model(base_outputs, training=False)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def make_heatmap_image(heatmap):
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(IMG_SIZE)

    heatmap_array = np.array(heatmap_img).astype(np.float32) / 255.0

    # Simple red-yellow heatmap without OpenCV or matplotlib
    color_heatmap = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    color_heatmap[..., 0] = np.uint8(255 * heatmap_array)
    color_heatmap[..., 1] = np.uint8(180 * heatmap_array)
    color_heatmap[..., 2] = 0

    return Image.fromarray(color_heatmap)

def overlay_heatmap(original_img_array, heatmap, alpha=0.45):
    original = Image.fromarray(original_img_array.astype(np.uint8)).convert("RGBA")

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE)
    heatmap_array = np.array(heatmap_img).astype(np.float32) / 255.0

    overlay_rgba = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 4), dtype=np.uint8)
    overlay_rgba[..., 0] = 255
    overlay_rgba[..., 1] = np.uint8(180 * heatmap_array)
    overlay_rgba[..., 2] = 0
    overlay_rgba[..., 3] = np.uint8(255 * alpha * heatmap_array)

    overlay = Image.fromarray(overlay_rgba).convert("RGBA")
    combined = Image.alpha_composite(original, overlay).convert("RGB")

    return combined

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("RSDKYAU Roadside Damage Classification")
st.write(
    "This web demo uses a fine-tuned DenseNet121 model to classify roadside "
    "infrastructure defects into four classes: Pot hole, crack, edge damage, and open drain."
)

st.sidebar.header("Model Information")
st.sidebar.write("Final Model: Fine-tuned DenseNet121")
st.sidebar.write("Test Accuracy: 97.48%")
st.sidebar.write("Weighted F1-score: 97.50%")
st.sidebar.write(f"Grad-CAM Layer: {gradcam_layer_name}")
st.sidebar.warning("This app performs image-level classification, not bounding-box object detection.")

uploaded_file = st.file_uploader(
    "Upload a roadside damage image",
    type=["jpg", "jpeg", "png"]
)

show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Analyzing image..."):
        pred_class, confidence, probs, img_array = predict_image(image)
        pred_idx = int(np.argmax(probs))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction Result")

        if confidence >= 0.70:
            st.success(f"Predicted Class: {pred_class}")
        else:
            st.warning(f"Low-confidence prediction: {pred_class}")

        st.info(f"Confidence: {confidence * 100:.2f}%")

        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probs
        })

        st.subheader("Class Probability")
        st.bar_chart(prob_df.set_index("Class"))
        st.dataframe(prob_df)

    if show_gradcam:
        st.subheader("Grad-CAM Explanation")

        if feature_extractor is None or classifier_model is None:
            st.error("Grad-CAM model could not be prepared.")
        else:
            with st.spinner("Generating Grad-CAM heatmap..."):
                heatmap = generate_gradcam(img_array, pred_index=pred_idx)

            if heatmap is None:
                st.error("Grad-CAM generation failed.")
            else:
                heatmap_img = make_heatmap_image(heatmap)
                overlay_img = overlay_heatmap(img_array, heatmap, alpha=0.45)

                g1, g2, g3 = st.columns(3)

                with g1:
                    st.caption("Original Image")
                    st.image(Image.fromarray(img_array.astype(np.uint8)), use_container_width=True)

                with g2:
                    st.caption("Grad-CAM Heatmap")
                    st.image(heatmap_img, use_container_width=True)

                with g3:
                    st.caption("Grad-CAM Overlay")
                    st.image(overlay_img, use_container_width=True)

                st.markdown(
                    """
                    **Grad-CAM interpretation:** Warmer highlighted regions indicate image areas that contributed more strongly to the model's prediction. 
                    This helps visually inspect whether the model is focusing on relevant road-damage regions rather than irrelevant background.
                    """
                )

else:
    st.info("Please upload a roadside damage image to classify.")