import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from src.config_v2 import NUM_CLASSES


def build_model():

    # Load pretrained ResNet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # -------------------------------------------------
    # Freeze all layers
    # -------------------------------------------------

    for param in model.parameters():
        param.requires_grad = False

    # -------------------------------------------------
    # Fine-tune only layer4
    # -------------------------------------------------

    for param in model.layer4.parameters():
        param.requires_grad = True

    # -------------------------------------------------
    # Replace classifier
    # -------------------------------------------------

    # Keep classifier identical to V1
    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    return model
