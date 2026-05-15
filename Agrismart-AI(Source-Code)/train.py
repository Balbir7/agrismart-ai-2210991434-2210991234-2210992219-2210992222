import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# --- 1. DATA PREPARATION ---
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('dataset/train', transform=data_transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

print("Classes used in training:")
print(train_data.classes)

# --- 2. MODEL SETUP ---
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))

# --- 3. TRAINING ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 10
print(f"\nStarting training for {epochs} epochs...")

for epoch in range(epochs):
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(train_loader):.4f}")

# --- 4. SAVE MODEL ---
if not os.path.exists('models'):
    os.makedirs('models')

torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': train_data.classes
}, 'models/best_model.pth')

print("\nTraining complete. Model saved successfully.")