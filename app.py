import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow import keras

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RSDKYAU Roadside Damage Classification",
    page_icon="🛣️",
    layout="wide"
)

# ============================================================
# BASIC CONFIG
# ============================================================

MODEL_PATH = "best_DenseNet121_finetuned.keras"

CLASS_NAMES = [
    "Pot hole",
    "crack_dataset",
    "edge_damage",
    "open_drain_dataset"
]

IMG_SIZE = (224, 224)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    try:
        model = keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )
        return model

    except Exception as e:
        st.error("Model loading failed.")
        st.exception(e)
        st.stop()


model = load_trained_model()

# ============================================================
# DEBUG HELPER
# ============================================================

def get_model_layer_summary():
    rows = []
    for i, layer in enumerate(model.layers):
        rows.append({
            "Index": i,
            "Layer Name": layer.name,
            "Layer Type": type(layer).__name__
        })
    return pd.DataFrame(rows)

# ============================================================
# PREPARE GRAD-CAM MODELS
# ============================================================

@st.cache_resource
def prepare_gradcam_models():
    try:
        # ----------------------------------------------------
        # Find nested Keras models inside the full model
        # Example: data_augmentation, densenet121
        # ----------------------------------------------------
        nested_models = [
            layer for layer in model.layers
            if isinstance(layer, keras.Model)
        ]

        if len(nested_models) == 0:
            return None, None, None, "No nested Keras model found inside saved model."

        candidate_models = []

        for nested_model in nested_models:
            conv_like_layer_count = 0
            total_layers = len(nested_model.layers)

            for sub_layer in nested_model.layers:
                try:
                    output_shape = sub_layer.output.shape
                    if len(output_shape) == 4:
                        conv_like_layer_count += 1
                except Exception:
                    pass

            candidate_models.append({
                "model": nested_model,
                "name": nested_model.name,
                "conv_like_layer_count": conv_like_layer_count,
                "total_layers": total_layers
            })

        # ----------------------------------------------------
        # Select the largest nested model with many 4D features
        # DenseNet121 will have many more layers than augmentation.
        # ----------------------------------------------------
        candidate_models = sorted(
            candidate_models,
            key=lambda x: (x["conv_like_layer_count"], x["total_layers"]),
            reverse=True
        )

        base_model = candidate_models[0]["model"]

        # Safety check: if selected model has too few layers, Grad-CAM may fail
        if len(base_model.layers) < 20:
            return None, None, None, (
                f"Selected nested model '{base_model.name}' has too few layers. "
                "DenseNet121 base model was not properly found."
            )

        # ----------------------------------------------------
        # Find last 4D convolutional feature layer
        # ----------------------------------------------------
        last_conv_layer = None

        for sub_layer in reversed(base_model.layers):
            try:
                output_shape = sub_layer.output.shape
                if len(output_shape) == 4:
                    last_conv_layer = sub_layer
                    break
            except Exception:
                continue

        if last_conv_layer is None:
            return None, None, None, "No suitable 4D convolutional feature layer found."

        # ----------------------------------------------------
        # Classifier head = layers after base_model in full model
        # ----------------------------------------------------
        try:
            base_index = model.layers.index(base_model)
        except ValueError:
            return None, None, None, "Base model was not found in full model layer list."

        classifier_layers = model.layers[base_index + 1:]

        if len(classifier_layers) == 0:
            return None, None, None, "No classifier head layers found after base model."

        # ----------------------------------------------------
        # Feature extractor
        # ----------------------------------------------------
        feature_extractor = keras.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, base_model.output]
        )

        # ----------------------------------------------------
        # Classifier model
        # ----------------------------------------------------
        classifier_input = keras.Input(shape=base_model.output.shape[1:])
        x = classifier_input

        for layer in classifier_layers:
            try:
                x = layer(x, training=False)
            except Exception:
                x = layer(x)

        classifier_model = keras.Model(classifier_input, x)

        layer_info = f"{base_model.name} → {last_conv_layer.name}"

        return feature_extractor, classifier_model, last_conv_layer.name, layer_info

    except Exception as e:
        return None, None, None, f"Grad-CAM preparation error: {str(e)}"


feature_extractor, classifier_model, gradcam_layer_name, gradcam_status = prepare_gradcam_models()

# ============================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================

def prepare_raw_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image).astype(np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    return img_array, img_batch


def predict_image(image):
    img_array, img_batch = prepare_raw_image(image)

    # The saved full model already contains preprocessing from training.
    # So we pass raw 0-255 RGB image batch directly.
    probs = model.predict(img_batch, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs, img_array, pred_idx

# ============================================================
# GRAD-CAM FUNCTIONS
# ============================================================

def generate_gradcam(img_array, pred_index=None):
    if feature_extractor is None or classifier_model is None:
        return None

    try:
        # Here we bypass the full model and directly use DenseNet base.
        # Therefore DenseNet preprocessing is needed here.
        input_tensor = np.expand_dims(img_array.copy(), axis=0)
        input_tensor = keras.applications.densenet.preprocess_input(input_tensor)

        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = feature_extractor(
                input_tensor,
                training=False
            )

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

    except Exception as e:
        st.error("Grad-CAM generation error.")
        st.exception(e)
        return None


def make_heatmap_image(heatmap):
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(IMG_SIZE)

    heatmap_array = np.array(heatmap_img).astype(np.float32) / 255.0

    color_heatmap = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)

    # Red-yellow heatmap without OpenCV/matplotlib
    color_heatmap[..., 0] = np.uint8(255 * heatmap_array)
    color_heatmap[..., 1] = np.uint8(180 * heatmap_array)
    color_heatmap[..., 2] = 0

    return Image.fromarray(color_heatmap)


def overlay_heatmap(original_img_array, heatmap, alpha=0.45):
    original = Image.fromarray(
        original_img_array.astype(np.uint8)
    ).convert("RGBA")

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

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("RSDKYAU Roadside Damage Classification")

st.markdown(
    """
    This web application uses a **fine-tuned DenseNet121** model to classify roadside
    infrastructure defects into four categories:

    **Pot hole, crack, edge damage, and open drain.**
    """
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Information")
st.sidebar.write("**Final Model:** Fine-tuned DenseNet121")
st.sidebar.write("**Test Accuracy:** 97.48%")
st.sidebar.write("**Weighted F1-score:** 97.50%")

if gradcam_layer_name is not None:
    st.sidebar.success("Grad-CAM is ready.")
    st.sidebar.write(f"**Grad-CAM Layer:** {gradcam_layer_name}")
    st.sidebar.caption(gradcam_status)
else:
    st.sidebar.error("Grad-CAM could not be prepared.")
    st.sidebar.caption(gradcam_status)

st.sidebar.warning(
    "This app performs image-level classification only. "
    "It does not provide bounding-box object detection."
)

with st.sidebar.expander("Model Layers Debug"):
    st.dataframe(get_model_layer_summary(), use_container_width=True)

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a roadside damage image",
    type=["jpg", "jpeg", "png"]
)

show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True)

# ============================================================
# MAIN APP
# ============================================================

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Analyzing image..."):
        pred_class, confidence, probs, img_array, pred_idx = predict_image(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction Result")

        if confidence >= 0.70:
            st.success(f"Predicted Class: {pred_class}")
        else:
            st.warning(f"Low-confidence Prediction: {pred_class}")

        st.info(f"Confidence: {confidence * 100:.2f}%")

        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probs
        })

        st.subheader("Class Probability")
        st.bar_chart(prob_df.set_index("Class"))
        st.dataframe(prob_df, use_container_width=True)

    # ========================================================
    # GRAD-CAM SECTION
    # ========================================================

    if show_gradcam:
        st.subheader("Grad-CAM Explanation")

        if feature_extractor is None or classifier_model is None:
            st.error("Grad-CAM could not be prepared for this model.")
            st.info(
                "Open the sidebar 'Model Layers Debug' section and check whether "
                "a DenseNet121 or large Functional model appears in the saved model."
            )
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
                    st.image(
                        Image.fromarray(img_array.astype(np.uint8)),
                        use_container_width=True
                    )

                with g2:
                    st.caption("Grad-CAM Heatmap")
                    st.image(heatmap_img, use_container_width=True)

                with g3:
                    st.caption("Grad-CAM Overlay")
                    st.image(overlay_img, use_container_width=True)

                st.markdown(
                    """
                    **Grad-CAM interpretation:**  
                    Warmer highlighted regions indicate the image areas that contributed
                    more strongly to the model's prediction. This helps visually inspect
                    whether the model is focusing on relevant road-damage regions rather
                    than irrelevant background.
                    """
                )

else:
    st.info("Please upload a roadside damage image to classify.")