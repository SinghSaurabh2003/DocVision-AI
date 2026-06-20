import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from src.config import NUM_CLASSES


def build_model():

    # Load pretrained ResNet-18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Number of input features to the final layer
    in_features = model.fc.in_features

    # Replace the classifier
    model.fc = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    return model