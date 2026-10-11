# --------------------------------------------------
# Streamlit application
# --------------------------------------------------

import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from data_utils import get_dataset_transform
from gradcam import create_gradcam
from model import build_pretrained_resnet18, get_device


MODEL_FILES = {
    "Feature extractor": os.path.join("models", "resnet18_feature_extractor.pth"),
    "Fine-tuned model": os.path.join("models", "resnet18_finetuned.pth"),
}


def load_model_metrics():
    """Load the saved training/test metrics for each model choice."""
    metrics_path = os.path.join("models", "training_metrics.json")
    if not os.path.exists(metrics_path):
        return {}

    with open(metrics_path, "r") as file:
        return json.load(file)


def get_class_names():
    """Detect the dataset class names from the train folder or from the root folder."""
    train_dir = os.path.join("data", "train")
    if os.path.isdir(train_dir):
        return sorted([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))])

    if os.path.isdir("data"):
        return sorted([name for name in os.listdir("data") if os.path.isdir(os.path.join("data", name))])

    return ["Class 0", "Class 1"]


def find_ground_truth_label(uploaded_image: Image.Image):
    """Return the class name for an uploaded image if it matches one in the dataset."""
    uploaded_array = np.asarray(uploaded_image.convert("RGB"))

    for root, _, files in os.walk("data"):
        for file_name in files:
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff")):
                continue

            candidate_path = os.path.join(root, file_name)
            try:
                candidate_image = Image.open(candidate_path).convert("RGB")
            except Exception:
                continue

            if np.array_equal(np.asarray(candidate_image), uploaded_array):
                return os.path.basename(os.path.dirname(candidate_path))

    return "Not found in dataset"


def resize_heatmap_for_display(heatmap, target_size):
    """Resize the Grad-CAM heatmap so it matches the original image dimensions."""
    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)
    heatmap_normalised = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
    heatmap_image = (heatmap_normalised * 255).astype(np.uint8)
    resized = Image.fromarray(heatmap_image).resize(target_size, resample=Image.BILINEAR)
    return np.asarray(resized)


def load_model_for_app(model_choice: str):
    """Load the selected trained model for prediction in the app."""
    model_path = MODEL_FILES[model_choice]
    if not os.path.exists(model_path):
        st.warning(f"No trained model was found for '{model_choice}'. Please run the training script first.")
        return None

    class_names = get_class_names()
    num_classes = len(class_names) if class_names else 2
    model = build_pretrained_resnet18(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=get_device()))
    model.eval()
    return model


def predict_single_image(model, image_path: str):
    """Predict the class label and Grad-CAM heatmap for a single input image."""
    image = Image.open(image_path).convert("RGB")
    transform = get_dataset_transform()
    tensor = transform(image).unsqueeze(0)

    device = get_device()
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = int(logits.argmax(dim=1).item())
        confidence = float(probabilities[0, predicted_index].item())

    gradcam = create_gradcam(model)
    heatmap = gradcam.generate_heatmap(tensor, predicted_index)

    return image, predicted_index, confidence, heatmap


def main():
    st.sidebar.header("Model selection")
    model_choice = st.sidebar.radio(
        "Choose the model to load:",
        ["Feature extractor", "Fine-tuned model"],
        help="The feature extractor keeps the pretrained ResNet18 backbone frozen and trains only the classifier head. The fine-tuned model continues training with a lower learning rate after unfreezing the backbone."
    )

    st.title("Chest X-ray Classification with Transfer Learning")
    st.markdown(
        """
        This app uses a pretrained ResNet18 model from ImageNet.

        In the first training stage, the ResNet18 backbone is frozen and reused as a feature extractor. Only the final classification head is trained for the chest X-ray task. This is a standard transfer-learning setup: the model already knows general visual patterns such as edges, textures, and shapes, and we adapt it to a new medical image problem.

        After that, the model can be fine-tuned more deeply if desired. The Grad-CAM heatmap highlights the image regions that most influenced the prediction, helping us explain why the model assigned a particular class.
        """
    )

    with st.container():
        st.markdown(
            """
            <div style='padding: 12px; border: 1px solid #ddd; border-radius: 8px; background-color: #f8f9fa;'>
            <strong>Training setup:</strong><br>
            • Backbone: ResNet18 pretrained on ImageNet<br>
            • Backbone weights: frozen<br>
            • Classifier head: trained<br>
            • Fine-tuning: optional after the head is trained
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        ### Quick learning guide

        - Feature extraction: the pretrained ResNet18 backbone acts like a fixed visual feature extractor. The early layers already know about edges, textures, and shapes.
        - Fine-tuning: after the head is trained, some deeper layers can be unfrozen and trained with a smaller learning rate on the medical dataset.
        - Grad-CAM: this method highlights the image regions that contributed most to the final prediction, helping us explain the model's decision.
        """
    )

    # Display the model performance metrics that were measured during training.
    st.subheader("Model performance")
    selected_model_path = MODEL_FILES[model_choice]
    metrics = load_model_metrics()
    model_label = "feature_extractor" if model_choice == "Feature extractor" else "fine_tuned"

    if os.path.exists(selected_model_path) and model_label in metrics:
        st.write("The current model was trained using transfer learning and then evaluated on the test set.")
        st.write("Training loss and accuracy are tracked during training, while the final test loss and test accuracy are reported after the full training loop.")

        train_loss = metrics[model_label]["train_loss"]
        train_accuracy = metrics[model_label]["train_accuracy"]
        test_loss = metrics[model_label]["test_loss"]
        test_accuracy = metrics[model_label]["test_accuracy"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train loss", f"{train_loss:.4f}")
        with col2:
            st.metric("Train acc", f"{train_accuracy * 100:.1f}%")
        with col3:
            st.metric("Test loss", f"{test_loss:.4f}")
        with col4:
            st.metric("Test acc", f"{test_accuracy * 100:.1f}%")
    else:
        st.info("Train the model first to populate the final performance metrics.")

    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        ground_truth_label = find_ground_truth_label(image)
        temp_path = os.path.join("temp_uploaded_image.png")
        image.save(temp_path)

        model = load_model_for_app(model_choice)
        if model is not None:
            image_result, predicted_index, confidence, heatmap = predict_single_image(model, temp_path)

            class_names = get_class_names()
            if not class_names:
                class_names = ["Class 0", "Class 1"]

            predicted_label = class_names[predicted_index]
            is_wrong_prediction = ground_truth_label != "Not found in dataset" and predicted_label != ground_truth_label

            st.subheader("Prediction")
            st.write(f"Ground truth label: {ground_truth_label}")
            if is_wrong_prediction:
                st.markdown(
                    f'<p style="color: red; font-weight: bold;">Predicted class: {predicted_label}</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.write(f"Predicted class: {predicted_label}")
            st.write(f"Confidence: {confidence:.2%}")

            col1, col2 = st.columns(2)
            with col1:
                st.image(image_result, caption=f"Uploaded image\nGround truth: {ground_truth_label}")

            with col2:
                resized_heatmap = resize_heatmap_for_display(heatmap, image_result.size)
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(image_result)
                ax.imshow(resized_heatmap, cmap="jet", alpha=0.45)
                ax.set_title("Grad-CAM heatmap")
                ax.axis("off")
                st.pyplot(fig)

    else:
        st.info("Upload an image from the dataset to see its ground-truth label and the Grad-CAM explanation.")


if __name__ == "__main__":
    main()
