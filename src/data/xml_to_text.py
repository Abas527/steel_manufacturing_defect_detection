import os
import xml.etree.ElementTree as ET
from PIL import Image

# Class mapping
CLASSES = [
    "normal",       # 0
    "crazing",      # 1
    "inclusion",    # 2
    "patches",      # 3
    "pitted_surface", # 4
    "rolled_in_scale", # 5
    "scratches"     # 6
]

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x_center * dw, y_center * dh, w * dw, h * dh)

def convert_annotations(img_root, ann_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all annotation files
    for xml_file in os.listdir(ann_dir):
        if not xml_file.endswith(".xml"):
            continue
        
        xml_path = os.path.join(ann_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_name = root.find("filename").text

        # Find image in any subfolder
        found_img_path = None
        for subfolder in os.listdir(img_root):
            img_path_candidate = os.path.join(img_root, subfolder, img_name)
            if os.path.exists(img_path_candidate):
                found_img_path = img_path_candidate
                break

        if found_img_path is None:
            print(f"Warning: Image {img_name} not found for {xml_file}")
            continue

        with Image.open(found_img_path) as img:
            w, h = img.size

        txt_filename = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, "w") as out_file:
            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in CLASSES:
                    continue
                cls_id = CLASSES.index(cls_name)

                xmlbox = obj.find("bndbox")
                xmin = float(xmlbox.find("xmin").text)
                xmax = float(xmlbox.find("xmax").text)
                ymin = float(xmlbox.find("ymin").text)
                ymax = float(xmlbox.find("ymax").text)

                bb = convert_bbox((w, h), (xmin, xmax, ymin, ymax))
                out_file.write(f"{cls_id} {' '.join(f'{a:.6f}' for a in bb)}\n")

        print(f"Converted: {txt_path}")

def create_empty_labels_for_normal(normal_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for img_file in os.listdir(normal_dir):
        if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            txt_filename = os.path.splitext(img_file)[0] + ".txt"
            open(os.path.join(output_dir, txt_filename), "w").close()

# Run for train
convert_annotations(
    img_root="data/train/images",
    ann_dir="data/train/annotations",
    output_dir="data/train/labels"
)
create_empty_labels_for_normal("data/train/images/normal", "data/train/labels")

# Run for validation
convert_annotations(
    img_root="data/valid/images",
    ann_dir="data/valid/annotations",
    output_dir="data/valid/labels"
)
create_empty_labels_for_normal("data/valid/images/normal", "data/valid/labels")
