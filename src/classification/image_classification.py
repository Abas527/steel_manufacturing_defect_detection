from pathlib import Path
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from torchvision import transforms,models
import torch
from torch import nn
import os
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import mlflow

train_dir= Path('data/train/images')
val_dir= Path('data/valid/images')
train_labels= Path('data/train/labels')
val_labels= Path('data/valid/labels')


def get_labels(label_file):
    if not os.path.exists(label_file) or os.path.getsize(label_file) == 0:
        return 0  
    with open(label_file, 'r') as f:
        classes = [int(line.strip().split()[0]) for line in f.readlines()]
        
        return 1 if any(cls != 0 for cls in classes) else 0

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

def prepare_dataset(data_dir, label_dir, transform):
    images = []
    labels = []

    for img_file in Path(data_dir).glob('*.jpg'):
        img = Image.open(img_file).convert('RGB')
        img = transform(img)
        images.append(img)
        
        label_file = Path(label_dir) / f"{img_file.stem}.txt"
        if os.path.exists(label_file):
            labels.append(get_labels(label_file))
        else:
            labels.append(0)  # default to normal if no label file
    
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

def loader(data_dir, label_dir, transform):
    images, labels = prepare_dataset(data_dir, label_dir, transform)
    dataset = torch.utils.data.TensorDataset(images, labels)
    loaders=torch.utils.data.DataLoader(dataset,batch_size=32, shuffle=True)
    return loaders
    
def get_model(num_classes):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):

    mlflow.set_experiment("Manufacturing defects Image_Classification_Experiment")

    with mlflow.start_run():

        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)



        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
        
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            mlflow.log_metric("loss", running_loss / len(train_loader), step=epoch)

             # Validation phase
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                predicted = []
                labels = []
                for images, label in val_loader:
                    images, label = images.to(device), label.to(device)
                    outputs = model(images)
                    _, prediction = torch.max(outputs.data, 1)
                    total += label.size(0)
                    correct += (prediction == label).sum().item()
                    predicted.extend(prediction.cpu().numpy())
                    labels.extend(label.cpu().numpy())

                
                model_summary={
                    'epoch': epoch + 1,
                    'loss': running_loss / len(train_loader),
                    'val_accuracy': 100 * correct / total,
                    "f1_score": f1_score(predicted, labels, average='weighted'),
                    "precision": precision_score(predicted, labels, average='weighted'),
                    "recall": recall_score(predicted, labels, average='weighted'),
                }
                mlflow.log_metrics(model_summary, step=epoch) 
    mlflow.log_artifact('model.pth')
    mlflow.pytorch.log_model(model, "model")
    return model

def main():
    num_classes = 2  
    train_loader = loader(train_dir, train_labels, transform)
    val_loader = loader(val_dir, val_labels, transform)

    model = get_model(num_classes)
    trained_model = train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'model.pth')
    print("Model saved as model.pth")

if __name__ == "__main__":
    main()