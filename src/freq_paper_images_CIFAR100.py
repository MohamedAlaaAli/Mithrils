import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
import csv
import sys
import os
import numpy as np
import random

from pathlib import Path

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call the seed-setting function
seed = 42  # You can choose any integer
set_seed(seed)

# Append the parent directory to sys.path to import your custom classes
current_file_path = Path().resolve()
sys.path.append(str(current_file_path.parent))

# Import your custom Dataset and ResNet34
from Utils.ImageDatasetHandler import Dataset
from Utils.ResNet import ResNet34

# Paths and parameters
save_dir = "../saved_models/cifar10"
pretrained_model_path = os.path.join(save_dir, "resnet34_cifar10_epoch50.pth")  # Path to the saved CIFAR-10 model
epochs = 50  # Maximum number of epochs for training on CIFAR-100
num_classes = 100  # Number of classes in CIFAR-100
early_stop_patience = 10  # Number of epochs to wait before early stopping
best_loss = float('inf')  # Initialize best loss as infinity
early_stop_counter = 0  # Initialize early stopping counter

# Initialize data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Use your custom Dataset class to load CIFAR-10 data
training_data = Dataset(data_dir="../data/CIFAR100/train", real=True, fourier=False, transform=transform)
test_data = Dataset(data_dir="../data/CIFAR100/test", real=True, fourier=False, transform=transform)

train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Load the pretrained ResNet34 model
model = ResNet34(num_classes=100)  # Initialize with the original number of classes (CIFAR-10)
model.load_state_dict(torch.load(pretrained_model_path))  # Load the CIFAR-10 trained weights

# Modify the final layer to output 100 classes (for CIFAR-100)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to('cuda')

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Function to print the learning rate
def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(f"Learning rate: {param_group['lr']:.6f}")


# File to save the results
results_file = "../Results/training_results_cifar100.csv"

# Write the header of the CSV file
with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Train F1-Score", "Test Loss", "Test Accuracy", "Test F1-Score"])

save_dir = "../saved_models/cifar100"

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    all_labels_train = []
    all_preds_train = []

    # Training phase
    model.train()
    for i, data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        all_labels_train.extend(labels.cpu().numpy())
        all_preds_train.extend(predicted.cpu().numpy())

    # Calculate training metrics
    f1_train = f1_score(all_labels_train, all_preds_train, average='weighted')
    average_train_loss = running_loss / len(train_loader)
    accuracy_train = 100 * correct_train / total_train

    # Evaluation phase
    model.eval()
    correct_test = 0
    total_test = 0
    running_test_loss = 0.0
    all_labels_test = []
    all_preds_test = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            all_labels_test.extend(labels.cpu().numpy())
            all_preds_test.extend(predicted.cpu().numpy())

    # Calculate test metrics
    f1_test = f1_score(all_labels_test, all_preds_test, average='weighted')
    average_test_loss = running_test_loss / len(test_loader)
    accuracy_test = 100 * correct_test / total_test

    # Print epoch metrics
    print("="*50)
    print(f"Epoch {epoch + 1}/{epochs}")
    print("-"*50)
    print(f"Train Loss:     {average_train_loss:.4f} | Train Accuracy:     {accuracy_train:.2f}% | Train F1-Score:     {f1_train:.2f}")
    print(f"Test Loss:      {average_test_loss:.4f}  | Test Accuracy:      {accuracy_test:.2f}% | Test F1-Score:      {f1_test:.2f}")
    print("="*50)
    print("\n")

    # Save the results to the CSV file
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, average_train_loss, accuracy_train, f1_train, average_test_loss, accuracy_test, f1_test])

    # Check for early stopping
    if average_test_loss < best_loss:
        best_loss = average_test_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), '../best_model_MNIST.pth')  # Save the best model
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    # Step the scheduler and print the learning rate
    scheduler.step(average_test_loss)
    print_lr(optimizer)

    # Save the model's state dictionary after training on CIFAR-100
    model_save_path = os.path.join(save_dir, f"resnet34_cifar100_epoch{epoch + 1}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
