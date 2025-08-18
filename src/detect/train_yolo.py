from ultralytics import YOLO
from pathlib import Path
import torch
from PIL import Image

def predict_yolo(image_path):
    yolo_model = YOLO("runs/detect/train/weights/best.pt")  # Load the trained YOLO model
    
    image = Image.open(image_path).convert("RGB")
    # find results
    results = yolo_model(image, conf=0.25, iou=0.45, agnostic_nms=True, max_det=1000)

    
    # results
    predictions = []
    for result in results:
        for box in result.boxes:
            predictions.append({
                'class': int(box.cls.item()),
                'confidence': box.conf.item(),
                'bbox': box.xyxy[0].tolist() 
            })
    
    return predictions

if __name__ == "__main__":
    image_path = Path("data/valid/images/crazing_243.jpg") 
    predictions = predict_yolo(image_path)
    
    for pred in predictions:
        print(f"Class: {pred['class']}, Confidence: {pred['confidence']}, BBox: {pred['bbox']}")