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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

# Append the parent directory to sys.path to import your custom classes
current_file_path = Path().resolve()
sys.path.append(str(current_file_path.parent))

# Import your custom Dataset and ResNet34
from Utils.ImageDatasetHandler import Dataset
from Utils.MobileNetv3 import MobileNetv3


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Initialize seed
seed = 42  # You can choose any integer value for the seed
set_seed(seed)

# Directory to save the model
save_dir = "../saved_models/mobilenet"
os.makedirs(save_dir, exist_ok=True)

# Define transformations
transform = transforms.Compose([
  # Resize CIFAR-10 images to 224x224
    transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
    transforms.RandomRotation(15),  # Random rotation within a range
    transforms.ToTensor(),
    transforms.Grayscale()
])


# Use your custom Dataset class to load CIFAR-10 data
training_data = Dataset(data_dir="../../data/MNIST/train", real=True, fourier=False, transform=transform)
test_data = Dataset(data_dir="../../data/MNIST/test", real=True, fourier=False, transform=transform)

# Create DataLoader
train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = MobileNetv3(in_channels=1, num_classes=10).to('cuda')
# model.load_state_dict(torch.load("../saved_models/mobilenet/mobilenet_cifar10_epoch99.pth"))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# Number of epochs
epochs = 100

# File to save the results
results_file = "../Results/MNIST_Mobnet_training_results.csv"

# Write the header of the CSV file
with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Epoch", "Train Loss", "Train Accuracy", "Train F1-Score", "Test Loss", "Test Accuracy", "Test F1-Score"])

# Training and evaluation loop
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

        # Calculate training accuracy and collect predictions/labels for F1-score
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        all_labels_train.extend(labels.cpu().numpy())
        all_preds_train.extend(predicted.cpu().numpy())

    # Calculate F1-score for training
    f1_train = f1_score(all_labels_train, all_preds_train, average='weighted')

    # Average training loss and accuracy
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

        scheduler.step(running_test_loss)  # Update the learning rate

    # Calculate F1-score for testing
    f1_test = f1_score(all_labels_test, all_preds_test, average='weighted')

    # Average test loss and accuracy
    average_test_loss = running_test_loss / len(test_loader)
    accuracy_test = 100 * correct_test / total_test

    # Improved print formatting
    print("\n")
    print("=" * 50)
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    print("-" * 50)
    print(
        f"Train Loss:     {average_train_loss:.4f} | Train Accuracy:     {accuracy_train:.2f}% | Train F1-Score:     {f1_train:.2f}")
    print(
        f"Test Loss:      {average_test_loss:.4f}  | Test Accuracy:      {accuracy_test:.2f}% | Test F1-Score:      {f1_test:.2f}")
    print("=" * 50)
    print("\n")

    # Save the results to the CSV file
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [epoch + 1, average_train_loss, accuracy_train, f1_train, average_test_loss, accuracy_test, f1_test])

    # Save the model's state dictionary
    if epoch <= 10:
        continue

    model_save_path = os.path.join(save_dir, f"mobilenet_MNIST_epoch{epoch}.pth")
    print(f"Model saved to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

