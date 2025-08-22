from flask import Flask, request, render_template, jsonify
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import io
import base64
import os

# Import torchvision models
from torchvision import models
import torch.nn as nn

app = Flask(__name__)

# Load your trained model
def load_model():
    # Recreate the same model architecture as in your training
    model = models.resnet18(pretrained=False)  # Don't need pretrained weights since we're loading your trained model
    model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes: empty, half-full, full
    
    # Load your trained weights
    model.load_state_dict(torch.load('fridge_resnet18.pth', map_location='cpu'))
    model.eval()
    return model

# Define image preprocessing (must match your training preprocessing)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Same as your training: 128x128
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Class labels
class_labels = ['full', 'half-full', 'empty']  # Adjust to match your classes

# Load model once when app starts
model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        # Get prediction results
        predicted_class = class_labels[predicted_class_idx]
        
        # Convert image to base64 for display
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2%}",
            'all_probabilities': {
                class_labels[i]: f"{probabilities[0][i].item():.2%}" 
                for i in range(len(class_labels))
            },
            'image': f"data:image/jpeg;base64,{image_base64}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)