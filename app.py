import streamlit as st
import torch
from PIL import Image
import tempfile

from src.utils import (
    load_model,
    predict_image,
    predict_topk
)

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Document Type Classifier",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Type Classifier")

st.write(
    """
Upload a document image and the trained ResNet18 model
will predict its document category.
"""
)

# ----------------------------------------------------
# Paths
# ----------------------------------------------------

CHECKPOINT_PATH = "checkpoints/best_model.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

@st.cache_resource
def get_model():

    model = load_model(
        CHECKPOINT_PATH,
        device
    )

    return model


model = get_model()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.title("Model Information")

st.sidebar.write("### Architecture")
st.sidebar.success("ResNet18")

st.sidebar.write("### Classes")
st.sidebar.success("9 Document Classes")

st.sidebar.write("### Transfer Learning")
st.sidebar.success("ImageNet Pretrained")

st.sidebar.write("### Framework")
st.sidebar.success("PyTorch")

st.sidebar.write("### Best Validation Accuracy")
st.sidebar.success("95.98%")

st.sidebar.write("### Device")
st.sidebar.info(device)

# ----------------------------------------------------
# Upload Image
# ----------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a document image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    # ---------------------------------------------
    # Save uploaded image temporarily
    # ---------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ) as tmp:

        image.save(tmp.name)

        image_path = tmp.name

    # ---------------------------------------------
    # Prediction Button
    # ---------------------------------------------

    if st.button("Predict Document Type"):

        with st.spinner("Predicting..."):

            prediction, confidence = predict_image(
                model,
                image_path,
                device
            )

            top3 = predict_topk(
                model,
                image_path,
                device,
                k=3
            )

        with col2:

            st.subheader("Prediction")

            st.success(prediction.capitalize())

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        st.markdown("---")

        st.subheader("🏆 Top 3 Predictions")

        for i, (cls, score) in enumerate(top3, start=1):

            st.write(
                f"**{i}. {cls.capitalize()}**"
            )

            st.progress(float(score))

            st.write(
                f"{score*100:.2f}%"
            )

        st.balloons()

# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <center>

    <h4>Document Type Classifier</h4>

    Built using ❤️ with PyTorch, ResNet18 and Streamlit

    </center>
    """,
    unsafe_allow_html=True
)
