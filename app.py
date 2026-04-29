import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageFilter
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

DISPLAY_NAMES = {
    "Pot hole": "Pothole",
    "crack_dataset": "Crack",
    "edge_damage": "Edge Damage",
    "open_drain_dataset": "Open Drain"
}

IMG_SIZE = (224, 224)

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: `{MODEL_PATH}`")
        st.stop()

    try:
        return keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )
    except Exception as e:
        st.error("Model loading failed.")
        st.exception(e)
        st.stop()


model = load_trained_model()


# ============================================================
# SAFE HELPERS
# ============================================================

def get_tensor_shape_safe(output_object):
    try:
        if isinstance(output_object, (list, tuple)):
            if len(output_object) == 0:
                return None
            output_object = output_object[0]

        if hasattr(output_object, "shape"):
            return output_object.shape

        return None
    except Exception:
        return None


def is_4d_output(layer):
    try:
        shape = get_tensor_shape_safe(layer.output)
        return shape is not None and len(shape) == 4
    except Exception:
        return False


def get_model_layer_summary():
    rows = []

    for i, layer in enumerate(model.layers):
        try:
            sub_layers = len(layer.layers) if isinstance(layer, keras.Model) else "-"
        except Exception:
            sub_layers = "-"

        rows.append({
            "Index": i,
            "Layer Name": layer.name,
            "Layer Type": type(layer).__name__,
            "Sub Layers": sub_layers
        })

    return pd.DataFrame(rows)


# ============================================================
# PREPARE GRAD-CAM
# ============================================================

@st.cache_resource
def prepare_gradcam_models():
    try:
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
                if is_4d_output(sub_layer):
                    conv_like_layer_count += 1

            candidate_models.append({
                "model": nested_model,
                "name": nested_model.name,
                "conv_like_layer_count": conv_like_layer_count,
                "total_layers": total_layers
            })

        candidate_models = sorted(
            candidate_models,
            key=lambda x: (x["conv_like_layer_count"], x["total_layers"]),
            reverse=True
        )

        debug_text = " | ".join([
            f"{c['name']} layers={c['total_layers']} conv4d={c['conv_like_layer_count']}"
            for c in candidate_models
        ])

        if candidate_models[0]["conv_like_layer_count"] == 0:
            return None, None, None, f"No convolution-like nested model found. Candidates: {debug_text}"

        base_model = candidate_models[0]["model"]

        all_4d_layers = []

        for sub_layer in base_model.layers:
            try:
                shape = get_tensor_shape_safe(sub_layer.output)

                if shape is not None and len(shape) == 4:
                    h = int(shape[1]) if shape[1] is not None else 0
                    w = int(shape[2]) if shape[2] is not None else 0

                    if h > 0 and w > 0:
                        all_4d_layers.append((sub_layer, h, w))
            except Exception:
                continue

        if len(all_4d_layers) == 0:
            return None, None, None, f"No suitable 4D feature layer found. Candidates: {debug_text}"

        # Prefer a 14x14 or larger feature map for clearer Grad-CAM.
        high_res_layers = [
            item for item in all_4d_layers
            if item[1] >= 14 and item[2] >= 14
        ]

        if len(high_res_layers) > 0:
            selected_layer, h, w = high_res_layers[-1]
        else:
            selected_layer, h, w = all_4d_layers[-1]

        selected_shape = f"{h}x{w}"
        last_conv_layer = selected_layer

        try:
            base_index = model.layers.index(base_model)
        except ValueError:
            return None, None, None, "Base model was not found inside the full model layer list."

        classifier_layers = model.layers[base_index + 1:]

        if len(classifier_layers) == 0:
            return None, None, None, "No classifier head layers found after base model."

        base_output = base_model.output
        if isinstance(base_output, (list, tuple)):
            base_output = base_output[0]

        feature_extractor = keras.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, base_output]
        )

        classifier_input_shape = tuple(base_output.shape[1:])
        classifier_input = keras.Input(shape=classifier_input_shape)
        x = classifier_input

        for layer in classifier_layers:
            try:
                x = layer(x, training=False)
            except Exception:
                x = layer(x)

        classifier_model = keras.Model(classifier_input, x)

        status = (
            f"Grad-CAM ready. Selected base model: {base_model.name}; "
            f"selected feature layer: {last_conv_layer.name}; "
            f"feature map size: {selected_shape}."
        )

        return feature_extractor, classifier_model, last_conv_layer.name, status

    except Exception as e:
        return None, None, None, f"Grad-CAM preparation error: {str(e)}"


feature_extractor, classifier_model, gradcam_layer_name, gradcam_status = prepare_gradcam_models()


# ============================================================
# IMAGE PROCESSING
# ============================================================

def prepare_raw_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image).astype(np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    return img_array, img_batch


def predict_image(image):
    img_array, img_batch = prepare_raw_image(image)

    # The saved full model already includes DenseNet preprocessing from training.
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
        input_tensor = np.expand_dims(img_array.copy(), axis=0)

        # We bypass the full model and use DenseNet base directly.
        input_tensor = keras.applications.densenet.preprocess_input(input_tensor)

        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = feature_extractor(
                input_tensor,
                training=False
            )

            tape.watch(conv_outputs)

            preds = classifier_model(base_outputs, training=False)

            if isinstance(preds, (list, tuple)):
                preds = preds[0]

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


def refine_heatmap(heatmap, threshold_percentile=55):
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    threshold = np.percentile(heatmap, threshold_percentile)
    heatmap = np.where(heatmap >= threshold, heatmap, 0)

    heatmap = np.power(heatmap, 0.8)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap


def make_heatmap_image(heatmap, threshold_percentile=55):
    heatmap = refine_heatmap(heatmap, threshold_percentile)

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(IMG_SIZE, resample=RESAMPLE_BICUBIC)
    heatmap_img = heatmap_img.filter(ImageFilter.GaussianBlur(radius=2))

    heatmap_array = np.array(heatmap_img).astype(np.float32) / 255.0

    color_heatmap = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)

    color_heatmap[..., 0] = np.uint8(255 * heatmap_array)

    green_component = np.maximum(heatmap_array - 0.25, 0) / 0.75
    color_heatmap[..., 1] = np.uint8(220 * green_component)

    color_heatmap[..., 2] = np.uint8(30 * (1 - heatmap_array) * (heatmap_array > 0))

    return Image.fromarray(color_heatmap)


def overlay_heatmap(original_img_array, heatmap, alpha=0.50, threshold_percentile=55):
    original = Image.fromarray(original_img_array.astype(np.uint8)).convert("RGBA")

    heatmap = refine_heatmap(heatmap, threshold_percentile)

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(IMG_SIZE, resample=RESAMPLE_BICUBIC)
    heatmap_img = heatmap_img.filter(ImageFilter.GaussianBlur(radius=2))

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
# REPORT HELPER
# ============================================================

def create_prediction_report(pred_class, confidence, probs):
    rows = []

    for cls, prob in zip(CLASS_NAMES, probs):
        rows.append({
            "Class": DISPLAY_NAMES[cls],
            "Probability": round(float(prob), 6),
            "Probability (%)": round(float(prob) * 100, 2)
        })

    df = pd.DataFrame(rows)

    report = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Final Model": "Fine-tuned DenseNet121",
        "Predicted Class": DISPLAY_NAMES[pred_class],
        "Confidence (%)": round(confidence * 100, 2),
        "Note": "Image-level classification only; not bounding-box object detection."
    }

    report_df = pd.DataFrame([report])

    return report_df, df


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("RSDKYAU Roadside Damage Classification")

st.markdown(
    """
    This web application demonstrates a **fine-tuned DenseNet121** model for
    image-level roadside infrastructure defect classification.

    The model classifies an uploaded image into one of four classes:
    **Pothole, Crack, Edge Damage, or Open Drain**.
    """
)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Final Model", "DenseNet121")
metric_col2.metric("Test Accuracy", "97.48%")
metric_col3.metric("Weighted F1-score", "97.50%")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.30,
    max_value=0.95,
    value=0.70,
    step=0.05
)

gradcam_alpha = st.sidebar.slider(
    "Grad-CAM overlay strength",
    min_value=0.20,
    max_value=0.80,
    value=0.50,
    step=0.05
)

heatmap_threshold = st.sidebar.slider(
    "Grad-CAM background suppression",
    min_value=30,
    max_value=80,
    value=55,
    step=5
)

st.sidebar.header("Model Information")
st.sidebar.write("**Final Model:** Fine-tuned DenseNet121")
st.sidebar.write("**Task:** Image-level classification")
st.sidebar.write("**Classes:** 4")
st.sidebar.warning(
    "This app does not draw bounding boxes. "
    "Grad-CAM is for visual explanation only, not pixel-level segmentation."
)

if gradcam_layer_name is not None:
    st.sidebar.success("Grad-CAM is ready.")
    st.sidebar.write(f"**Grad-CAM Layer:** {gradcam_layer_name}")
    st.sidebar.caption(gradcam_status)
else:
    st.sidebar.error("Grad-CAM could not be prepared.")
    st.sidebar.caption(gradcam_status)

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

    sorted_indices = np.argsort(probs)[::-1]
    top1_idx = int(sorted_indices[0])
    top2_idx = int(sorted_indices[1])

    top1_class = CLASS_NAMES[top1_idx]
    top2_class = CLASS_NAMES[top2_idx]

    top1_prob = float(probs[top1_idx])
    top2_prob = float(probs[top2_idx])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction Result")

        display_pred_class = DISPLAY_NAMES[pred_class]

        if confidence >= confidence_threshold:
            st.success(f"Predicted Class: {display_pred_class}")
        else:
            st.warning(f"Low-confidence Prediction: {display_pred_class}")

        st.info(f"Confidence: {confidence * 100:.2f}%")

        t1, t2 = st.columns(2)
        t1.metric("Top-1 Prediction", DISPLAY_NAMES[top1_class], f"{top1_prob * 100:.2f}%")
        t2.metric("Top-2 Prediction", DISPLAY_NAMES[top2_class], f"{top2_prob * 100:.2f}%")

        prob_df = pd.DataFrame({
            "Class": [DISPLAY_NAMES[c] for c in CLASS_NAMES],
            "Probability": probs,
            "Probability (%)": probs * 100
        })

        prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)

        st.subheader("Class Probability")
        st.bar_chart(prob_df.set_index("Class")["Probability"])
        st.dataframe(
            prob_df.style.format({
                "Probability": "{:.4f}",
                "Probability (%)": "{:.2f}%"
            }),
            use_container_width=True
        )

    report_df, prob_report_df = create_prediction_report(pred_class, confidence, probs)

    report_csv = pd.concat(
        [
            report_df,
            pd.DataFrame([{}]),
            prob_report_df
        ],
        ignore_index=True
    ).to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Report CSV",
        data=report_csv,
        file_name="rsdkyau_prediction_report.csv",
        mime="text/csv"
    )

    if show_gradcam:
        st.subheader("Grad-CAM Explanation")

        if feature_extractor is None or classifier_model is None:
            st.error("Grad-CAM could not be prepared for this model.")
        else:
            with st.spinner("Generating Grad-CAM heatmap..."):
                heatmap = generate_gradcam(img_array, pred_index=pred_idx)

            if heatmap is None:
                st.error("Grad-CAM generation failed.")
            else:
                heatmap_img = make_heatmap_image(
                    heatmap,
                    threshold_percentile=heatmap_threshold
                )

                overlay_img = overlay_heatmap(
                    img_array,
                    heatmap,
                    alpha=gradcam_alpha,
                    threshold_percentile=heatmap_threshold
                )

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
                    The highlighted warm regions indicate image areas that contributed most strongly
                    to the model's prediction. The visualization is expected to focus on visible
                    defect-related regions such as pothole surfaces, cracks, damaged road edges,
                    or open-drain structures. This explanation supports visual interpretability
                    but does **not** represent pixel-level segmentation.
                    """
                )

else:
    st.info("Please upload a roadside damage image to classify.")