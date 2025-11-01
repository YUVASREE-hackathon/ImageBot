import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import os

# ✅ Define class labels based on MySQL categories
CLASS_LABELS = ["African Bush Elephant", "African Forest Elephant", "Asian Elephant"]

# ✅ Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ Load dataset (Make sure dataset is labeled properly)
dataset = ImageFolder(root="elephant_data", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ✅ Load pretrained ResNet model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_LABELS))  # Adjust for 3 types
for param in model.fc.parameters():
    param.requires_grad = True  # Train only the last layer specific elephant types

# ✅ Define optimizer & loss function
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate
NUM_EPOCHS = 25  # Increase training epochs

# ✅ Train the model
NUM_EPOCHS = 38
for epoch in range(NUM_EPOCHS):
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}")

# ✅ Save the trained model
torch.save(model.state_dict(), "elephant_model.pth")
print("✅ Model training complete!")