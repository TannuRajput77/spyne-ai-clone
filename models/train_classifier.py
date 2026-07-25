import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Paths
DATA_DIR = "D:/Projects/spyne-ai-clone/datasets/angle_dataset_split"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image transforms (resize + normalize for ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_data = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform)
valid_data = datasets.ImageFolder(f"{DATA_DIR}/valid", transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)

print("Classes:", train_data.classes)   # should print ['front', 'rear', 'side']

# Load pretrained ResNet18 (lighter/faster than ResNet34, good for CPU)
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Train Acc: {train_acc:.2f}%")

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"           Validation Acc: {val_acc:.2f}%")

# Save model
torch.save(model.state_dict(), "D:/Projects/spyne-ai-clone/models/angle_classifier.pt")
print("\nModel saved to models/angle_classifier.pt")