# --------------------------------------------------
# Model definition
# --------------------------------------------------

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_pretrained_resnet18(num_classes: int) -> nn.Module:
    """Create a ResNet18 model with ImageNet weights and a new classifier head.

    This is transfer learning. The network already knows how to detect basic visual
    patterns such as edges, textures and shapes from ImageNet. We reuse those learned
    features and replace the final layer so it can classify chest X-rays.
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # The original classifier layer expects 1000 ImageNet classes.
    # We replace it with a smaller classifier that matches our dataset.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def freeze_feature_layers(model: nn.Module) -> None:
    """Freeze the convolutional base so only the classifier learns new weights."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze only the final classification layer.
    # This is the key transfer-learning step.
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True


def unfreeze_all_layers(model: nn.Module) -> None:
    """Unfreeze the full network for fine-tuning after the classifier is trained."""
    for parameter in model.parameters():
        parameter.requires_grad = True


def get_device() -> torch.device:
    """Return the best available device: GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model_for_training(num_classes: int, freeze_feature_extractor: bool = True):
    """Create the model and optionally freeze early layers."""
    model = build_pretrained_resnet18(num_classes)

    if freeze_feature_extractor:
        freeze_feature_layers(model)

    return model.to(get_device())
