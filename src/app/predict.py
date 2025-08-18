# predict.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

model_path = Path('model.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes=["Non Deffective", "Defective"]

def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    model = models.resnet18(weights=None)
    num_classes = 2  
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def pre_processing(image):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  
    return image.to(device)

def predict(model_path, image_path):
    image = pre_processing(image_path)
    model=load_model(model_path)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()