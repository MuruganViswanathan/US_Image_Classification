# image_train_validation.py

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Directory for your dataset
data_dir = 'S:/Engineering/AI Datasets/PresetFreeImaging/Test3/'

# Image transformations: Resize and normalize images to fit GoogLeNet input requirements
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to GoogLeNet input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize to ImageNet means/std
])

# Load dataset from the image folder
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
class_names = dataset.classes  # Get class names from folder structure
num_classes = len(class_names)  # Number of classes

print(f"class_names: {class_names}, num_classes = {num_classes}")

# Split dataset into training and validation (70% training, 30% validation)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"train_size = {train_size}, val_size = {val_size}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

# Load pre-trained GoogLeNet
net = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)

# Modify the final layer for our dataset's number of classes
net.fc = nn.Linear(net.fc.in_features, num_classes)

# Set up training
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.SGD(net.parameters(),
                      lr=3e-4,
                      momentum=0.9,
                      weight_decay=1e-4)

# Transfer model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")
net = net.to(device)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=6):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        # Training loop
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

            # Track progress
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f'Batch {i+1}/{len(train_loader)} - Loss: {running_loss / (i+1):.4f} - Accuracy: {100 * correct / total:.2f}%')

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # Collect labels and predictions for confusion matrix
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        # Epoch progress
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1} completed in {epoch_time:.2f} seconds.')
        print(f'Training Loss: {running_loss / len(train_loader):.4f} | Training Accuracy: {100 * correct / total:.2f}%')
        print(f'Validation Loss: {val_loss / len(val_loader):.4f} | Validation Accuracy: {100 * val_correct / val_total:.2f}%')

    return all_labels, all_preds

# Confusion Matrix Plotting
def plot_confusion_matrix_with_percentages(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)

    # Recall per class (handling division by zero)
    recall = np.diag(cm) / np.where(cm.sum(axis=1) != 0, cm.sum(axis=1), 1) * 100

    # Precision per class (handling division by zero)
    precision = np.diag(cm) / np.where(cm.sum(axis=0) != 0, cm.sum(axis=0), 1) * 100

    # Create the confusion matrix with totals
    cm_with_totals = np.vstack([np.hstack([cm, cm.sum(axis=1, keepdims=True)]),
                                np.hstack([cm.sum(axis=0), [np.sum(np.diag(cm))]])])

    # Fill in the last column with precision and the last row with recall
    cm_with_totals[:-1, -1] = precision  # Last column for precision
    cm_with_totals[-1, :-1] = recall  # Last row for recall
    cm_with_totals[-1, -1] = np.sum(np.diag(cm)) / np.sum(cm) * 100  # Overall accuracy in last cell

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_with_totals, annot=True, fmt='.0f', cmap='Blues', cbar=False,
                xticklabels=class_names + ['Recall (%)'],
                yticklabels=class_names + ['Precision (%)'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix with Class-wise Precision and Recall')
    plt.show()


# Train the model and plot confusion matrix
all_labels, all_preds = train_model(net, train_loader, val_loader, criterion, optimizer)
plot_confusion_matrix_with_percentages(all_labels, all_preds, class_names)

# Save the trained model
torch.save(net.state_dict(), 'googlenet_ultrasound1.pth')
print('Model saved as googlenet_ultrasound.pth')
