import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm  # NEW: For progress bar

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 24
NUM_CLASSES = 9 # Adjust based on your gesture classes
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train_split_aug")
VAL_DIR = os.path.join(DATA_DIR, "val_split")
MODEL_PATH = "gesture_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transforms
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Datasets
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform_train)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CNN Model (ResNet18)
model = models.resnet18(weights="IMAGENET1K_V1")  # Use pretrained weights
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# For stats
history = {"epoch": [], "loss": [], "train_acc": [], "val_acc": []}

def validate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

def train():
    best_val_acc = 0.0
    best_model_path = "gesture_cnn_best.pth"
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
            for images, labels in pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                running_loss = total_loss / (total_samples / BATCH_SIZE)
                running_acc = 100 * total_correct / total_samples
                pbar.set_postfix({"loss": f"{running_loss:.4f}", "acc": f"{running_acc:.2f}%"})

            train_acc = 100 * total_correct / total_samples
            val_acc = validate()
            avg_loss = total_loss / len(train_loader)
            print(f"\nEpoch {epoch+1} Summary: Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%\n")
            history["epoch"].append(epoch+1)
            history["loss"].append(avg_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"\U0001F389 New best model saved with Val Acc: {best_val_acc:.2f}% at '{best_model_path}'")
    except KeyboardInterrupt:
        print("Training interrupted by user, saving model...")
        torch.save(model.state_dict(), MODEL_PATH)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Model saved as '{MODEL_PATH}'")
    # Print summary table
    print("\nTraining Summary:")
    print(f"{'Epoch':<6}{'Loss':<12}{'Train Acc':<12}{'Val Acc':<12}")
    for i in range(len(history["epoch"])):
        print(f"{history['epoch'][i]:<6}{history['loss'][i]:<12.4f}{history['train_acc'][i]:<12.2f}{history['val_acc'][i]:<12.2f}")

if __name__ == "__main__":
    train()
