# --------------------------------------------------
# Data loading utilities
# --------------------------------------------------

import os
import shutil
from typing import Tuple, List

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def get_dataset_transform() -> transforms.Compose:
    """Return the image transforms used for ResNet18 training and testing."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def detect_class_folders(data_root: str) -> List[str]:
    """Return the class names found in the dataset directory."""
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Dataset folder not found: {data_root}")

    class_folders = sorted(
        [name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))]
    )

    if not class_folders:
        raise ValueError(f"No class folders were found inside: {data_root}")

    return class_folders


def build_train_val_test_folders(data_root: str) -> Tuple[str, str, str]:
    """Create the standard three split folders if the dataset is in a single folder.

    The original images are not moved or copied. This function simply organises the
    class directories into train/val/test by creating split directories and returning
    their paths.
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
        return train_dir, val_dir, test_dir

    classes = detect_class_folders(data_root)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in classes:
        class_dir = os.path.join(data_root, class_name)
        image_files = [
            os.path.join(class_dir, name)
            for name in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, name))
        ]

        if len(image_files) < 3:
            raise ValueError(
                f"Not enough images in class '{class_name}' to create a train/val/test split."
            )

        train_files, temp_files = train_test_split(image_files, test_size=0.4, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

        for split_name, file_list in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ]:
            split_class_dir = os.path.join(data_root, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for image_path in file_list:
                image_name = os.path.basename(image_path)
                destination = os.path.join(split_class_dir, image_name)
                if not os.path.exists(destination):
                    shutil.copy2(image_path, destination)

    return train_dir, val_dir, test_dir


def get_data_directories(data_root: str) -> Tuple[str, str, str]:
    """Return train/val/test directories for the dataset.

    If the dataset already contains train/val/test folders we use them. If not, we
    create a train/val/test split from a single parent folder containing class folders.
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
        return train_dir, val_dir, test_dir

    if os.path.isdir(data_root):
        class_directories = detect_class_folders(data_root)
        if class_directories and not any(
            os.path.isdir(os.path.join(data_root, split_name))
            for split_name in ("train", "val", "test")
        ):
            return build_train_val_test_folders(data_root)

    raise FileNotFoundError(
        "The dataset should contain train/val/test folders or a single class directory layout."
    )


def get_train_val_test_loaders(data_root: str, batch_size: int = BATCH_SIZE):
    """Build DataLoaders for the chest X-ray dataset."""
    train_dir, val_dir, test_dir = get_data_directories(data_root)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=get_dataset_transform())
    val_dataset = datasets.ImageFolder(root=val_dir, transform=get_dataset_transform())
    test_dataset = datasets.ImageFolder(root=test_dir, transform=get_dataset_transform())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes


def get_single_image_loader(image_path: str):
    """Create a loader for one image for the Streamlit app."""
    transform = get_dataset_transform()
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image
