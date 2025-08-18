import os
import shutil
from pathlib import Path
import yaml

# Paths to your dataset
BASE_DIR = Path("data")
TRAIN_IMG_DIR = BASE_DIR / "train" / "images"
TRAIN_LABEL_DIR = BASE_DIR / "train" / "labels"
VAL_IMG_DIR = BASE_DIR / "valid" / "images"
VAL_LABEL_DIR = BASE_DIR / "valid" / "labels"

# Class names in the correct YOLO order
CLASSES = [
    "normal", 
    "crazing", 
    "inclusion", 
    "patches", 
    "pitted_surface", 
    "rolled_in_scale", 
    "scratches"
]

def flatten_images_and_labels(img_root, label_root):
    """
    Moves images from class subfolders into a single images/ folder,
    keeps matching .txt files in labels/ folder.
    """
    img_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)

    # Go through each class folder
    for class_folder in img_root.glob("*"):
        if class_folder.is_dir():
            for img_path in class_folder.glob("*.*"):
                new_img_path = img_root / img_path.name
                shutil.move(str(img_path), str(new_img_path))

                # Matching label file
                label_path = label_root.parent / "annotation" / (img_path.stem + ".txt")
                if label_path.exists():
                    shutil.move(str(label_path), str(label_root / label_path.name))

            # Remove empty folder
            try:
                class_folder.rmdir()
            except OSError:
                pass

def create_data_yaml(output_path):
    """
    Creates YOLO data.yaml file
    """
    data_yaml = {
        "path": str(BASE_DIR),
        "train": "train/images",
        "val": "valid/images",
        "names": {i: name for i, name in enumerate(CLASSES)}
    }

    with open(output_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

if __name__ == "__main__":
    print("Flattening train images & labels...")
    flatten_images_and_labels(TRAIN_IMG_DIR, TRAIN_LABEL_DIR)

    print("Flattening validation images & labels...")
    flatten_images_and_labels(VAL_IMG_DIR, VAL_LABEL_DIR)

    print("Creating data.yaml...")
    create_data_yaml(BASE_DIR / "data.yaml")

    print("✅ Dataset prepared for YOLO training!")
