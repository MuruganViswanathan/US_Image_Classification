import os
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch  # For softmax
import matplotlib.pyplot as plt

# Define the organ classes based on folder names
classes = ['Breast', 'Kidney', 'Liver', 'Ovary', 'Spleen', 'Thyroid', 'Uterus']

# Define transformation to resize and normalize the images
transform = transforms.Compose([
    transforms.Resize(224),                   # Resize shorter side to 256 pixels
    transforms.CenterCrop(224),               # Crop the center to get 224x224 images
    transforms.ToTensor(),                    # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load the ONNX model
model_path = '../presetFreeImaging_googlenet_1.onnx'
session = ort.InferenceSession(model_path)

# Get model input and output names
input_name = session.get_inputs()[0].name  # Input name (should be 'data')
output_name = session.get_outputs()[0].name  # Output name (should be 'prob')

print(f"input_name = {input_name}, output_name = {output_name}")

# Function to apply softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Shift values to avoid overflow
    return exp_x / exp_x.sum()

# Function to preprocess and classify an image
def classify_image(image_path):
    # Open image and apply transformations
    image = Image.open(image_path).convert('RGB')  # Ensure it's 3-channel RGB
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    input_data = input_tensor.numpy()  # Convert to numpy array

    # Run the model and get the prediction
    result = session.run([output_name], {input_name: input_data})
    predictions = np.array(result).squeeze()

    # Apply softmax to convert logits to probabilities
    probabilities = softmax(predictions)

    # Get the predicted class (index of highest probability)
    predicted_class = np.argmax(probabilities)

    # Debug: Print top 3 predicted classes with their probabilities
    top_predictions = np.argsort(probabilities)[-3:][::-1]
    for idx in top_predictions:
        print(f"Class {idx} ({classes[idx]}): {probabilities[idx]:.4f}")

    return predicted_class


# Prepare lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Root folder containing the organ subfolders
image_folder = '../../Images'

# Iterate through each organ class folder
for class_idx, organ in enumerate(classes):
    organ_folder = os.path.join(image_folder, organ)

    # Iterate through all images in this folder
    for image_name in os.listdir(organ_folder):
        image_path = os.path.join(organ_folder, image_name)

        if image_name.endswith(('.JPG', '.jpg', '.jpeg')):
            # Classify the image and store results
            predicted_class = classify_image(image_path)
            predicted_labels.append(predicted_class)
            true_labels.append(class_idx)
            print(f"{image_path:<80}: \t true_class = {class_idx},\t predicted_class = {predicted_class}")

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)

# Show the plot
plt.title('Confusion Matrix')
plt.show()
