"""
Streamlit Web Application for Plant Leaf Disease Detection & Diagnostics.

Features:
- Cached model loading via @st.cache_resource
- Top-3 predictions with probability bars
- Interactive Grad-CAM visual explanation overlay
- Actionable agronomic treatment & prevention recommendations
- Out-of-distribution / low-confidence warnings (< 60%)
- Error handling for invalid/corrupt image uploads
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import streamlit as st

# Add project root to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils import CLASS_NAMES, format_class_name, load_image_for_inference
from src.gradcam import make_gradcam_heatmap, overlay_gradcam, find_last_conv_layer

# Page Configuration
st.set_page_config(
    page_title="Plant Leaf Disease Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI
st.markdown("""
<style>
    .reportview-container {
        background-color: #f8fafc;
    }
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #10b981;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading deep learning model...")
def load_trained_model():
    """
    Load the trained Keras model once and cache it in memory.
    Checks candidate model paths in models/ directory.
    """
    import tensorflow as tf

    candidate_paths = [
        os.path.join(BASE_DIR, "models", "mobilenetv2_best.keras"),
        os.path.join(BASE_DIR, "models", "baseline_best.keras"),
        os.path.join(BASE_DIR, "models", "trained_plant_disease_model.keras"),
        os.path.join(BASE_DIR, "trained_plant_disease_model.keras")
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                return model, os.path.basename(path)
            except Exception as e:
                st.warning(f"Failed to load model from {path}: {e}")

    return None, None


@st.cache_data
def load_disease_database():
    """Load agronomic disease tips and prevention database."""
    info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "disease_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Initialize Model & Info Database
model, model_filename = load_trained_model()
disease_db = load_disease_database()

# Sidebar
st.sidebar.image(
    "https://img.icons8.com/fluency/96/plant-under-sun.png",
    width=80
)
st.sidebar.title("🌿 Plant Diagnostics")
st.sidebar.caption("Portfolio-grade Deep Learning Diagnostic Assistant")
page = st.sidebar.radio(
    "Navigation",
    ["Disease Recognition", "About Dataset & System", "Model Architecture"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Deployment Status")
if model is not None:
    st.sidebar.success(f"Active Model: `{model_filename}`")
else:
    st.sidebar.warning("No `.keras` model found in `models/`.\n\nPlease train a model or download pre-trained weights.")

st.sidebar.markdown("""
> **Note for HF Spaces / Streamlit Cloud:**  
> Large model weights (>100MB) should be tracked with **Git LFS** or downloaded from Hugging Face Hub during startup.
""")


# ==========================================
# PAGE 1: DISEASE RECOGNITION (MAIN)
# ==========================================
if page == "Disease Recognition":
    st.title("Plant Leaf Disease Recognition")
    st.markdown(
        "Upload a close-up image of a crop leaf on a neutral background to identify disease conditions and view explainability heatmaps."
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Select a leaf photo (JPG, PNG, JPEG)",
            type=["jpg", "jpeg", "png"],
            help="Images from the New Plant Diseases Dataset are single leaves on plain background resized to 128x128."
        )

        show_cam = st.checkbox("Generate Grad-CAM Explainability Heatmap", value=True)
        cam_alpha = st.slider("Grad-CAM Overlay Intensity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

        if uploaded_file is not None:
            try:
                pil_image = Image.open(uploaded_file).convert("RGB")
                st.image(pil_image, caption="Uploaded Leaf Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error opening image file: {e}")
                pil_image = None
        else:
            st.info("👆 Please upload a leaf image to begin diagnostic inference.")
            pil_image = None

    with col2:
        st.subheader("2. Diagnostic Results")
        if uploaded_file is not None and pil_image is not None:
            if model is None:
                st.error("No model is loaded. Train a model with `python src/train.py` or place a `.keras` file in `models/`.")
            else:
                with st.spinner("Analyzing foliar patterns..."):
                    img_arr, resized_img = load_image_for_inference(pil_image, target_size=(128, 128))
                    preds = model.predict(img_arr, verbose=0)[0]
                    top_indices = np.argsort(preds)[::-1][:3]

                    top_class_id = top_indices[0]
                    top_class_name = CLASS_NAMES[top_class_id]
                    top_confidence = float(preds[top_class_id])
                    top_formatted = format_class_name(top_class_name)

                # Low-confidence Warning
                if top_confidence < 0.60:
                    st.warning(
                        f"⚠️ **Low Confidence Prediction ({top_confidence * 100:.1f}%)**: "
                        "The image may be out-of-distribution (e.g. full plant, cluttered background, non-leaf object, or unrepresented disease). Please interpret with caution."
                    )
                else:
                    st.success(f"**Primary Diagnosis: {top_formatted}** (Confidence: {top_confidence * 100:.1f}%)")

                # Top-3 Predictions
                st.markdown("#### Top Predictions")
                for rank, idx in enumerate(top_indices, 1):
                    raw_name = CLASS_NAMES[idx]
                    formatted = format_class_name(raw_name)
                    prob = float(preds[idx])
                    st.write(f"**{rank}. {formatted}** — `{prob * 100:.2f}%`")
                    st.progress(min(prob, 1.0))

                # Grad-CAM Visualization
                if show_cam:
                    st.markdown("#### Explainability: Grad-CAM Saliency")
                    try:
                        heatmap = make_gradcam_heatmap(img_arr, model, pred_index=top_class_id)
                        overlay_img = overlay_gradcam(pil_image, heatmap, alpha=cam_alpha)
                        st.image(
                            overlay_img,
                            caption=f"Grad-CAM Attention Heatmap for {top_formatted}",
                            use_column_width=True
                        )
                        st.caption("Warm regions (red/yellow) indicate the specific leaf tissue patterns driving the model's prediction.")
                    except Exception as e:
                        st.info(f"Could not compute Grad-CAM for this model architecture: {e}")

                # Agronomic Recommendations
                st.markdown("---")
                st.markdown("#### 🌾 Treatment & Prevention Recommendations")
                info = disease_db.get(top_class_name, None)
                if info:
                    with st.expander(f"Recommended Actions for {info.get('condition', top_formatted)}", expanded=True):
                        st.markdown(f"**Observed Symptoms:**\n{info.get('symptoms', 'N/A')}")
                        st.markdown(f"**Recommended Treatment:**\n{info.get('treatment', 'N/A')}")
                        st.markdown(f"**Long-term Prevention:**\n{info.get('prevention', 'N/A')}")
                else:
                    st.info("No specific agronomic treatment record registered for this condition.")

        else:
            st.write("Awaiting image upload to produce diagnosis.")


# ==========================================
# PAGE 2: ABOUT DATASET & SYSTEM
# ==========================================
elif page == "About Dataset & System":
    st.title("About Plant Leaf Disease Detection")

    st.markdown("""
    ### Dataset Profile
    - **Source**: *New Plant Diseases Dataset* (re-engineered from the original PlantVillage benchmark).
    - **Total Images**: 87,000+ laboratory leaf images across **38 classes** (healthy and diseased states of apple, corn, grape, potato, tomato, and more).
    - **Image Modality**: Single leaves photographed on standardized plain/indoor backgrounds, resized to **128×128 pixels**.
    - **Important Caveat**: These are controlled lab-bench images, not field or aerial photos. Performance on wild in-field imagery may degrade without domain adaptation.

    ### Data Leakage Note
    The original offline-augmented dataset contains multiple augmented copies of identical leaves. If split randomly, augmented copies appear in both train and validation sets, inflating validation accuracy.
    In this repository, we introduce a **source-grouped split** that isolates unique leaf identities to ensure true held-out test evaluation.

    ### 38 Supported Categories
    """)

    classes_df = pd.DataFrame({
        "Index": range(len(CLASS_NAMES)),
        "Raw Class Identifier": CLASS_NAMES,
        "Clean Class Name": [format_class_name(c) for c in CLASS_NAMES]
    })
    st.dataframe(classes_df, use_container_width=True, hide_index=True)


# ==========================================
# PAGE 3: MODEL ARCHITECTURE
# ==========================================
elif page == "Model Architecture":
    st.title("Model Architecture & Benchmarks")

    st.markdown("""
    ### Evaluated Architectures

    1. **Improved Baseline CNN**:
       - 5 Convolutional blocks (32, 64, 128, 256, 512 filters)
       - `BatchNormalization` after every convolution layer to stabilize learning
       - `Dropout` (0.20 to 0.40) to mitigate overparameterization
       - On-the-fly random flip, rotation, zoom, and contrast augmentation

    2. **MobileNetV2 Transfer Learning**:
       - Pre-trained on ImageNet-1K with depthwise-separable convolutions
       - 2-Phase training: frozen-base feature extractor followed by fine-tuning top 30% layers
       - Optimized for edge and mobile deployment with low latency

    3. **EfficientNetB0 Transfer Learning**:
       - Pre-trained backbone with compound scaling (depth, width, and resolution)

    ### Edge Deployment (TensorFlow Lite)
    Trained models can be exported to `.tflite` format for edge devices (Raspberry Pi, Android, iOS):
    ```bash
    python src/utils.py --export-tflite models/mobilenetv2_best.keras
    ```
    """)
