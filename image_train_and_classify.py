# image_train_validation.py
# Author: Murugan Viswanathan
# Given a data_dir with Images grouped into subfolders with the class names
# this program loads the images from dir, transforms the images, and gets class
# names using folder names. Then, splits the dataset into train/val sets and trains
# a GoogleNet model using the train dataset. Then it uses the val dataset to
# plots various validation curves and confusion matrix to help in fine-tuning.
# The trained model is stored in an ONNX file. Then the ONNX file is validated again
# using val dataset.

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
import onnx
import onnxruntime as ort

# Directory for your dataset
data_dir = 'S:/Engineering/AI Datasets/PresetFreeImaging/Test3/'
# data_dir = '../../Images/Test3/'

#########################################################################################
# Load dataset from dir, transform, get class names using folder names
def prepare_dataset(data_dir):
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
    return dataset, class_names, num_classes


#########################################################################################
def split_dataset(dataset):
    # Split dataset into training and validation (70% training, 30% validation)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"train_size = {train_size}, val_size = {val_size}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader


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
# Training function with learning curve tracking and batch-wise progress
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=9):
    total_time = 0  # total training time for all epochs

    num_images = len(train_loader.dataset)  # Unique number of training images
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []  # Track learning curves

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')

        # Training phase
        model.train()  # train mode
        running_loss = 0.0
        running_correct = 0
        total_train = 0

        start_time = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            running_correct += predicted.eq(labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f'Batch {i + 1}/{len(train_loader)} - Train Loss: {running_loss / (i + 1):.4f}, '
                      f'Train Accuracy: {100 * running_correct / total_train:.2f}%')

        # Validation phase (once per epoch)
        model.eval()  # Evaluation mode, so that the weights are not changed

        val_loss, val_correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
                _, val_predicted = val_outputs.max(1)
                total_val += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()

        # Record losses and accuracies for plotting
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = 100 * running_correct / total_train
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = 100 * val_correct / total_val
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_accuracy)
        val_accuracies.append(epoch_val_accuracy)

        # Track epoch time
        epoch_time = time.time() - start_time  # train + validation
        total_time += epoch_time
        print(f'Epoch {epoch + 1} completed in {epoch_time:.2f} seconds.')
        print(f'Training Loss: {epoch_train_loss:.4f} | Training Accuracy: {epoch_train_accuracy:.2f}%')
        print(f'Validation Loss: {epoch_val_loss:.4f} | Validation Accuracy: {epoch_val_accuracy:.2f}%')

    print(f'\n=====> Total Training Time for {len(train_loader.dataset)} images for all {num_epochs} epochs = {total_time:.2f} seconds.\n')

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

    # Plot learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.show()

    return final_labels, final_preds


#########################################################################################
# Confusion Matrix Plotting
def plot_confusion_matrix_with_percentages(labels, preds, class_names):
    # print("Plotting Confusion matrix")
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


#########################################################################################
## Prepare dataset, Train the model and Classify/Validata results
#########################################################################################
# dataset
dataset, class_names, num_classes = prepare_dataset(data_dir)
train_dataset, val_dataset, train_loader, val_loader = split_dataset(dataset)

# Load pre-trained GoogLeNet
net = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
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

# Save the model as ONNX format with dynamic batch size
dummy_input = torch.randn(1, 3, 224, 224, device=device)  # Ensure correct NCHW format
torch.onnx.export(
    net,
    dummy_input,
    "googlenet_ultrasound.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11,  # Use a stable ONNX opset version
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("Model saved as googlenet_ultrasound.onnx with dynamic batch size.")


# Validation using ONNX model
def validate_onnx_model(val_loader):
    ort_session = ort.InferenceSession("googlenet_ultrasound.onnx")
    onnx_labels, onnx_preds = [], []

    for inputs, labels in val_loader:
        inputs = inputs.numpy()  # Convert to NumPy
        inputs = np.transpose(inputs, (0, 2, 3, 1))  # Ensure NCHW format if needed
        ort_inputs = {ort_session.get_inputs()[0].name: inputs}
        ort_outs = ort_session.run(None, ort_inputs)
        predicted = np.argmax(ort_outs[0], axis=1)
        onnx_labels.extend(labels.numpy())
        onnx_preds.extend(predicted)

    # Plot confusion matrix
    plot_confusion_matrix_with_percentages(onnx_labels, onnx_preds, class_names)



print("\nValidation using ONNX model..")
validate_onnx_model(val_loader)
print("done.\n")


