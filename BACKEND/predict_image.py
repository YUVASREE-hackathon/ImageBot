import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# ✅ Load the trained model
model = models.resnet50(weights=None)  # Ensure it's the same architecture used during training
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # Ensure it has 3 output classes
model.load_state_dict(torch.load("elephant_model.pth", map_location=torch.device("cpu")))
model.eval()

# ✅ Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ Load and preprocess the image
image_path = "1.jpg"  # Change this if needed
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# ✅ Make a prediction
with torch.no_grad():
    output = model(image)
    predicted_class = output.argmax(dim=1).item()

# ✅ Define class labels
class_labels = ["African Bush Elephant", "African Forest Elephant", "Asian Elephant"]

# ✅ Print the result
print(f"Predicted Species: {class_labels[predicted_class]}")

print(model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']))
