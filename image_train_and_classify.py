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
from collections import Counter

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


#########################################################################################
# Function to count instances in each class for train and validation sets
def count_class_instances(dataset, indices, class_names):
    targets = np.array(dataset.targets)  # Get all class labels for the full dataset
    target_subset = targets[indices]  # Get labels for the specific subset (train/val)
    class_counts = Counter(target_subset)  # Count instances of each class
    print("Class distribution:")
    for idx, class_name in enumerate(class_names):
        print(f"{class_name}: {class_counts.get(idx, 0)}")


#########################################################################################
# Training function
# Updated training function to collect predictions only once after final epoch
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=6):
    total_time = 0
    num_images = len(train_loader.dataset)  # Unique number of training images

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        # Training loop
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f'Batch {i + 1}/{len(train_loader)} - Loss: {running_loss / (i + 1):.4f} - Accuracy: {100 * correct / total:.2f}%')

        # Track epoch time
        epoch_time = time.time() - start_time
        total_time += epoch_time
        print(f'Epoch {epoch + 1} completed in {epoch_time:.2f} seconds.')
        print(f'Training Loss: {running_loss / len(train_loader):.4f} | Training Accuracy: {100 * correct / total:.2f}%')

    # Print total training time
    print(f'\n===> Total Training Time for {num_images} images across {num_epochs} epochs = {total_time:.2f} seconds.\n')

    # Final validation phase after all epochs
    print("Final validation using validation dataset...")
    final_labels = []  # Labels for confusion matrix at final validation
    final_preds = []  # Predictions for confusion matrix at final validation
    model.eval()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            final_labels.extend(labels.cpu().numpy())
            final_preds.extend(predicted.cpu().numpy())


    return final_labels, final_preds


#########################################################################################
# Confusion Matrix Plotting
def plot_confusion_matrix_with_percentages(labels, preds, class_names):
    print("Plotting Confusion matrix")
    cm = confusion_matrix(labels, preds)

    # Last Row: Precision per class (handling division by zero)
    precision = np.diag(cm) / np.where(cm.sum(axis=0) != 0, cm.sum(axis=0), 1) * 100

    # Last Column: Recall per class (handling division by zero)
    recall = np.diag(cm) / np.where(cm.sum(axis=1) != 0, cm.sum(axis=1), 1) * 100

    # Create the confusion matrix with totals
    cm_with_totals = np.vstack([np.hstack([cm, recall.reshape(-1, 1)]),  # Recall in last column
                                np.hstack([precision, [np.sum(np.diag(cm)) / np.sum(cm) * 100]])])  # Precision in last row

    # Convert the matrix to strings
    cm_str = cm_with_totals.astype(str)

    # Add '%' signs to the last row and last column
    cm_str[:-1, -1] = [f'{val:.2f}%' for val in cm_with_totals[:-1, -1]]
    cm_str[-1, :-1] = [f'{val:.2f}%' for val in cm_with_totals[-1, :-1]]
    cm_str[-1, -1] = f'{cm_with_totals[-1, -1]:.2f}%'  # Overall accuracy

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_with_totals, annot=cm_str, fmt='', cmap='Blues', cbar=False,
                xticklabels=class_names + ['Recall (%)'],
                yticklabels=class_names + ['Precision (%)'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix with Class-wise Precision and Recall')
    plt.show()


#########################################################################################################

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
print(f"\n===> device = {device}\n")
net = net.to(device)

# Print class distributions for training and validation sets
print("\nTraining data class distribution:")
count_class_instances(dataset, train_dataset.indices, class_names)
print("\nValidation data class distribution:")
count_class_instances(dataset, val_dataset.indices, class_names)

# Train the model and plot confusion matrix
final_labels, final_preds = train_model(net, train_loader, val_loader, criterion, optimizer)
plot_confusion_matrix_with_percentages(final_labels, final_preds, class_names)

# Save the trained model
torch.save(net.state_dict(), 'googlenet_ultrasound.pth')
print('Model saved as googlenet_ultrasound.pth')
