# --------------------------------------------------
# Training and evaluation utilities
# --------------------------------------------------

import os
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(model: nn.Module, data_loader: DataLoader, optimizer, criterion, device: torch.device):
    """Train the model for one epoch.

    This function is intentionally simple and easy to read so students can see the
    steps involved in a standard PyTorch training loop.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion, device: torch.device):
    """Evaluate the model on a validation or test set."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def save_model(model: nn.Module, file_path: str) -> None:
    """Save the trained model weights."""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(model.state_dict(), file_path)


def load_model(model: nn.Module, file_path: str) -> nn.Module:
    """Load model weights from disk."""
    model.load_state_dict(torch.load(file_path, map_location="cpu"))
    return model
