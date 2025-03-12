### Author: Murugan Viswanathan
### Image inference using the Trained PTH file
###  Classifies a specified image into one of the 'class_names'
###  It loads a pretrained 'googlenet_ultrasound.pth' image located in the same folder and
###  and uses it for doing the inference.
###     Usage: python classify_image.py <path_to_image>

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os
import time

# Class labels (adjust these to match your dataset structure)
class_names = ['Breast', 'Kidney', 'Liver', 'Ovary', 'Spleen', 'Thyroid', 'Uterus']

# Define image transformation to match the model's input requirements
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

start_time = time.time()

# Set GPU cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained trained googlenet_ultrasound.pth file
model = models.googlenet(weights=None, aux_logits=False)  # Start with a non-pretrained GoogLeNet model and disable aux classifiers
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Adjust final layer
model.load_state_dict(torch.load('googlenet_ultrasound.pth', map_location=device))  # Load saved weights
model = model.to(device)  # load model to GPU if available
model.eval()  # Set model to evaluation mode

load_time = time.time() - start_time
print(f'\nLoaded pretrained googlenet_ultrasound.pth in = {load_time:.2f} seconds.\n')

# Function to predict the class of a single image
def classify_image(image_path):
    start_time = time.time()

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

    end_time = time.time() - start_time

    # Print the predicted label
    print(f"Predicted Label = {class_names[predicted.item()]}")
    print(f'\nTime taken = {end_time:.2f} seconds.\n')

######################################################################################
# Read image file path from command line argument
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python classify_image.py <path_to_image>")
    else:
        image_path = sys.argv[1]
        classify_image(image_path)
