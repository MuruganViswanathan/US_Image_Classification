## Author: Murugan Viswanathan
## Image inference using the Trained PTH file

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

# Class labels (adjust these to match your dataset structure)
class_names = ['Breast', 'Kidney', 'Liver', 'Ovary', 'Spleen', 'Thyroid', 'Uterus']

# Define image transformation to match the model's input requirements
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.googlenet(weights=None, aux_logits=False)  # Start with a non-pretrained GoogLeNet model and disable aux classifiers
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Adjust final layer
model.load_state_dict(torch.load('googlenet_ultrasound.pth', map_location=device))  # Load saved weights
model = model.to(device)  # load model to GPU if available
model.eval()  # Set model to evaluation mode

# Function to predict the class of a single image
def classify_image(image_path):
    # Check if file exists
    if not os.path.isfile(image_path):
        print(f"File {image_path} not found.")
        return

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image).unsqueeze(0)  # Add batch dimension

    # Move image to GPU if available
    image = image.to(device)

    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Print the predicted label
    print(f"Predicted Label: {class_names[predicted.item()]}")

######################################################################################
# Read image file path from command line argument
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python classify_image.py <path_to_image>")
    else:
        image_path = sys.argv[1]
        classify_image(image_path)
