# --------------------------------------------------
# Training script
# --------------------------------------------------

import argparse
import json
import os

import torch
from torch import nn
from torch.optim import Adam

from data_utils import get_train_val_test_loaders
from model import create_model_for_training
from training_utils import evaluate_model, save_model, train_one_epoch


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a chest X-ray classifier using transfer learning.")
    parser.add_argument("--data-root", type=str, default="data", help="Path to the dataset folder.")
    parser.add_argument("--epochs-head", type=int, default=5, help="Number of epochs to train the new classifier head.")
    parser.add_argument("--epochs-finetune", type=int, default=3, help="Number of epochs for fine-tuning the whole model.")
    parser.add_argument("--learning-rate-head", type=float, default=1e-3, help="Learning rate for the classifier head.")
    parser.add_argument("--learning-rate-finetune", type=float, default=1e-4, help="Learning rate during fine-tuning.")
    return parser.parse_args()


def save_metrics(metrics):
    """Save model performance metrics as JSON so the app can display the right values."""
    os.makedirs("models", exist_ok=True)
    metrics_path = os.path.join("models", "training_metrics.json")
    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=2)


def main():
    args = parse_arguments()

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(
            f"Dataset folder not found: {args.data_root}. "
            "Place the chest X-ray data inside the 'data/' folder before training."
        )

    train_loader, val_loader, test_loader, classes = get_train_val_test_loaders(args.data_root)
    num_classes = len(classes)

    print(f"Found classes: {classes}")
    print(f"Number of classes: {num_classes}")

    metrics = {}

    # --------------------------------------------------
    # Step 1: feature extraction with a frozen backbone
    # --------------------------------------------------
    # ResNet18 was trained on ImageNet. That means it already knows useful visual
    # features such as edges, corners, textures and simple shapes.
    #
    # In transfer learning, we often freeze the early convolutional layers and only
    # train the final classifier. This is a good first step because the pre-trained
    # backbone already contains useful knowledge and we only need to adapt it to the
    # chest X-ray problem.
    model = create_model_for_training(num_classes=num_classes, freeze_feature_extractor=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Only the new classifier head is trainable in this stage.
    # The backbone is frozen, so its learned ImageNet representations are kept.
    optimizer = Adam(model.fc.parameters(), lr=args.learning_rate_head)

    print("Training the new classifier while the pre-trained base remains frozen.")
    for epoch in range(args.epochs_head):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{args.epochs_head} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")

    save_model(model, os.path.join("models", "resnet18_feature_extractor.pth"))
    feature_extractor_test_loss, feature_extractor_test_accuracy = evaluate_model(model, test_loader, criterion, device)
    metrics["feature_extractor"] = {
        "train_loss": round(float(train_loss), 4),
        "train_accuracy": round(float(train_acc), 4),
        "val_loss": round(float(val_loss), 4),
        "val_accuracy": round(float(val_acc), 4),
        "test_loss": round(float(feature_extractor_test_loss), 4),
        "test_accuracy": round(float(feature_extractor_test_accuracy), 4),
    }

    # --------------------------------------------------
    # Step 2: fine-tuning the whole model
    # --------------------------------------------------
    # Now we unfreeze the entire network and continue learning with a lower learning rate.
    # This is called fine-tuning: the model starts from the previously learned features
    # and adjusts them slightly to the new chest X-ray classification task.
    #
    # We use a lower learning rate because the pre-trained weights are already useful.
    # A large learning rate could overwrite these valuable features too quickly.
    print("Fine-tuning the full model with a lower learning rate.")
    for parameter in model.parameters():
        parameter.requires_grad = True

    optimizer = Adam(model.parameters(), lr=args.learning_rate_finetune)

    for epoch in range(args.epochs_finetune):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"Fine-tune {epoch + 1}/{args.epochs_finetune} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")

    save_model(model, os.path.join("models", "resnet18_finetuned.pth"))
    fine_tuned_test_loss, fine_tuned_test_accuracy = evaluate_model(model, test_loader, criterion, device)
    metrics["fine_tuned"] = {
        "train_loss": round(float(train_loss), 4),
        "train_accuracy": round(float(train_acc), 4),
        "val_loss": round(float(val_loss), 4),
        "val_accuracy": round(float(val_acc), 4),
        "test_loss": round(float(fine_tuned_test_loss), 4),
        "test_accuracy": round(float(fine_tuned_test_accuracy), 4),
    }

    save_metrics(metrics)
    print(f"Final feature-extractor test loss: {metrics['feature_extractor']['test_loss']:.4f}")
    print(f"Final feature-extractor test accuracy: {metrics['feature_extractor']['test_accuracy']:.4f}")
    print(f"Final fine-tuned test loss: {metrics['fine_tuned']['test_loss']:.4f}")
    print(f"Final fine-tuned test accuracy: {metrics['fine_tuned']['test_accuracy']:.4f}")
    print("Training finished. Model weights and metrics saved in the models/ folder.")


if __name__ == "__main__":
    main()
