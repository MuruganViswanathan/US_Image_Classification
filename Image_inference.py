import torch
from torchvision import models, transforms
from PIL import Image
import argparse

# Define image transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
num_classes = 7  # Update based on your dataset
net = models.googlenet(pretrained=False, aux_logits=False)  # Disable auxiliary classifiers
net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
net.load_state_dict(torch.load('googlenet_ultrasound.pth'))
net.eval()  # Set model to evaluation mode

# Transfer to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

# Function to preprocess the image and run inference
def predict_image(image_path):
    image = Image.open(image_path)
    image = data_transforms(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs = net(image)
        _, predicted = outputs.max(1)

    return predicted.item()

# Command-line argument parser
def main():
    parser = argparse.ArgumentParser(description='Ultrasound Image Inference')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    # Predict class for the input image
    predicted_class = predict_image(args.image_path)
    print(f'Predicted class: {predicted_class}')

if __name__ == "__main__":
    main()
