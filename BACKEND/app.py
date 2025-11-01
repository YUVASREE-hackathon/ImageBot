from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torchvision.models as models
import openai
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Replace with your actual OpenAI API key
openai.api_key = "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"

# Load Trained Elephant Model (For Image Analysis)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 3)

try:
    model.load_state_dict(torch.load("elephant_model.pth", map_location=torch.device("cpu")))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

# Image Preprocessing Function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# API: Analyze Elephant Image
@app.route("/analyze_elephant", methods=["POST"])
def analyze_elephant():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file uploaded or empty file"}), 400

    try:
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax(dim=1).item()

    class_labels = ["African Bush Elephant", "African Forest Elephant", "Asian Elephant"]
    species = class_labels[predicted_class]

    return jsonify({"species": species})

# API: ChatGPT-Powered Elephant Query System
@app.route("/elephant_chat", methods=["POST"])
def elephant_chat():
    data = request.get_json()
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    system_prompt = (
        "You are an AI assistant specializing in elephants. "
        "Answer only elephant-related questions, including their habitat, diet, size, population, conservation status, etc. "
        "If a user asks something unrelated, politely decline to answer."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )

        chatbot_reply = response["choices"][0]["message"]["content"]
        return jsonify({"reply": chatbot_reply})

    except Exception as e:
        return jsonify({"error": f"Chatbot error: {str(e)}"}), 500

# Run Flask App on port 5001
if __name__ == "__main__":
    app.run(debug=True, port=5001)
