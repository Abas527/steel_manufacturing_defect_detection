import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
import io

# --- Model loading utility ---
def load_model(model_path, num_classes=2, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

# --- Preprocessing utility ---
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

#I don't know how to implement GRAD-CAM so i take help of captum library
# https://captum.ai/tutorials/Grad_Cam_ResNet

# --- Grad-CAM computation ---
def get_gradcam(model, input_tensor, target_layer=None, target_class=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    if target_layer is None:
        # Default to last conv layer for ResNet18
        target_layer = model.layer4[1].conv2

    gradcam = LayerGradCam(model, target_layer)
    if target_class is None:
        # Use predicted class if not specified
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()
    attr = gradcam.attribute(input_tensor, target=target_class)
    # Upsample to input size
    attr_upsampled = torch.nn.functional.interpolate(attr, size=(224, 224), mode='bilinear', align_corners=False)
    return attr_upsampled.squeeze().cpu().detach().numpy(), target_class

# --- Overlay heatmap on image ---
def overlay_heatmap_on_image(orig_img: Image.Image, cam_mask: np.ndarray, alpha=0.5):
    # Normalize cam_mask to [0, 1]
    cam_mask = (cam_mask - cam_mask.min()) / (cam_mask.max() - cam_mask.min() + 1e-8)
    cam_mask = np.uint8(255 * cam_mask)
    cam_mask = Image.fromarray(cam_mask).resize(orig_img.size, resample=Image.BILINEAR).convert("L")
    cam_mask = np.array(cam_mask)

    # Create color heatmap
    cmap = plt.get_cmap('jet')
    heatmap = cmap(cam_mask / 255.0)[:, :, :3]  # Drop alpha channel
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap).resize(orig_img.size)

    # Overlay
    overlayed = Image.blend(orig_img.convert("RGB"), heatmap_img, alpha=alpha)
    return overlayed

# --- Main Grad-CAM pipeline for Streamlit ---
def gradcam_on_image(image: Image.Image, model_path: str, num_classes=2, device=None):
    model = load_model(model_path, num_classes=num_classes, device=device)
    input_tensor = preprocess_image(image)
    cam_mask, pred_class = get_gradcam(model, input_tensor, device=device)
    overlayed_img = overlay_heatmap_on_image(image, cam_mask)
    return overlayed_img, pred_class