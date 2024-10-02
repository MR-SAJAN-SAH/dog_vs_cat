import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import gdown

# Define the CNN model (same architecture as used during training)
gdown.download('https://drive.google.com/file/d/1BJf2SBr9383z-WXXLkc1R2_tmXSSjJhA/view?usp=sharing', quiet=False)
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
        x = self.pool(nn.ReLU()(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(nn.ReLU()(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = self.pool(nn.ReLU()(self.conv3(x)))  # Conv3 -> ReLU -> Pool
        x = x.view(-1, 128 * 16 * 16)  # Flatten the output
        x = nn.ReLU()(self.fc1(x))  # Fully connected layer
        x = self.fc2(x)  # Output layer
        return x

# Create the model object
model = CNN()

# Load the saved model weights
model.load_state_dict(torch.load('cat_dog_classifier.pth'))

# Set the model to evaluation mode (important for inference)
model.eval()

# Define the image transformation (same as used during training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to the size expected by the model
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize
])

# Function to predict the class of a new image
def predict_image(image_path, model):
    # Open the image, apply the transformations
    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    # Make the prediction
    output = model(img)
    _, predicted = torch.max(output, 1)  # Get the index of the highest probability

    # Convert predicted index to class (0 for cat, 1 for dog)
    if predicted.item() == 0:
        print("It's a dog!")
    else:
        print("It's a cat!")

# Example usage with a new image
predict_image("m.jpeg", model)
