import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
import os

# ✅ Define Image Transformations (Move this Above train_dataset)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ Define Dataset Path (Update If Needed)
dataset_path = "E:/IMAGERECOGNITIONCHATBOT/BACKEND/elephant_data"

# ✅ Load Dataset with Transform
train_dataset = ImageFolder(root=dataset_path, transform=transform)

# ✅ Count Images in Each Category
bush_elephant_count = len(os.listdir(os.path.join(dataset_path, "African Bush Elephant")))
forest_elephant_count = len(os.listdir(os.path.join(dataset_path, "African Forest Elephant")))
asian_elephant_count = len(os.listdir(os.path.join(dataset_path, "Asian Elephant")))

# ✅ Compute Class Weights (Fixing Incorrect Calculation)
class_counts = [bush_elephant_count, forest_elephant_count, asian_elephant_count]
total_samples = sum(class_counts)
weights = [total_samples / count if count > 0 else 0 for count in class_counts]  # Avoid division by zero

print(f"Class Counts: {class_counts}")
print(f"Class Weights: {weights}")

# ✅ Use Weighted Sampling
sample_weights = [weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

# ✅ Load Pretrained ResNet50 Model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 Elephant Species

# ✅ Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Train the Model
NUM_EPOCHS = 38
for epoch in range(NUM_EPOCHS):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}")

# ✅ Save the Trained Model
torch.save(model.state_dict(), "elephant_model.pth")
print("✅ Model training complete and saved as 'elephant_model.pth'")
