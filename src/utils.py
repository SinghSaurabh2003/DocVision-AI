import torch
import torchvision.transforms as transforms

from PIL import Image

from src.model_v2 import build_model

from src.config_v2 import (
    IMAGE_SIZE,
    NUM_CLASSES,
)

# -------------------------------------
# Class Names
# -------------------------------------

CLASS_NAMES = [

    "advertisement",
    "budget",
    "email",
    "filefolder",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo"

]

# -------------------------------------
# Image Transform
# -------------------------------------

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])

# -------------------------------------
# Load Trained Model
# -------------------------------------

def load_model(checkpoint_path, device):

    model = build_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model


# -------------------------------------
# Predict Image
# -------------------------------------

def predict_image(
    model,
    image_path,
    device
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = CLASS_NAMES[
        prediction.item()
    ]

    confidence = confidence.item()

    return predicted_class, confidence


# -------------------------------------
# Top K Prediction
# -------------------------------------

def predict_topk(
    model,
    image_path,
    device,
    k=3
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, indices = torch.topk(
            probabilities,
            k
        )

    predictions = []

    for score, idx in zip(
        confidence[0],
        indices[0]
    ):

        predictions.append(

            (
                CLASS_NAMES[idx.item()],
                score.item()
            )

        )

    return predictions
