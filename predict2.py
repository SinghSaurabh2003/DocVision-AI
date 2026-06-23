import os
import torch

from src.utils import (
    load_model,
    predict_image,
    predict_topk
)

# =====================================================
# PATHS
# =====================================================

# Change this according to your environment

CHECKPOINT_PATH = "/kaggle/working/Document-Type-Classifier/checkpoints/best_model.pth"

IMAGE_PATH = "/kaggle/input/datasets/singhsaurabh03/testing/test/invoice/invoice0.tif"

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print(f"Using Device : {device}")
print("=" * 60)

# =====================================================
# CHECK FILES
# =====================================================

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading Model...")

model = load_model(
    CHECKPOINT_PATH,
    device
)

print("Model Loaded Successfully!")

# =====================================================
# SINGLE PREDICTION
# =====================================================

prediction, confidence = predict_image(
    model,
    IMAGE_PATH,
    device
)

print("\n")

print("=" * 60)
print("Prediction")
print("=" * 60)

print(f"Image Path      : {IMAGE_PATH}")
print(f"Prediction      : {prediction}")
print(f"Confidence      : {confidence*100:.2f}%")

# =====================================================
# TOP 3 PREDICTIONS
# =====================================================

print("\n")

print("=" * 60)
print("Top 3 Predictions")
print("=" * 60)

top3 = predict_topk(
    model,
    IMAGE_PATH,
    device,
    k=3
)

for i, (cls, score) in enumerate(top3, start=1):

    print(
        f"{i}. {cls:<20} {score*100:.2f}%"
    )

print("\n")

print("=" * 60)
print("Prediction Completed")
print("=" * 60)
