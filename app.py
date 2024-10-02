import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify
import io

# Define the CNN model (same as the one used during training)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model
model = CNN()
model.load_state_dict(torch.load('cat_dog_classifier.pth'))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Initialize Flask application
app = Flask(__name__)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img = transform(img).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            class_name = 'cat' if predicted.item() == 0 else 'dog'
        
        return jsonify({'prediction': class_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
