import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, random_split
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
    transforms.Resize((224, 224)),  # Resize CIFAR-10 images to 224x224
    transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
    transforms.RandomRotation(15),  # Random rotation within a range
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Use your custom Dataset class to load CIFAR-10 data
training_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Split the training dataset into train and validation sets
train_size = int(0.8 * len(training_data))
val_size = len(training_data) - train_size
train_dataset, val_dataset = random_split(training_data, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Load the pre-trained MobileNetV3 from torch.hub
# Initialize MobileNetV3 model
model = models.mobilenet_v3_large(pretrained=False)

# Modify the classifier for CIFAR-100 and add Dropout
num_classes = 100
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, 1280),
    nn.Hardswish(),
    nn.Dropout(p=0.5),  # Dropout layer added
    nn.Linear(1280, num_classes)
)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Load the saved model weights
model.load_state_dict(torch.load('../saved_models/mobilenet/mobilenet_cifar100v3_torch_epoch22.pth', map_location=device))
# Freeze all layers except the classifier
# for param in model.parameters():
#     param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True


# Loss, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=.3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Number of epochs
epochs = 100

# File to save the results
results_file = "../Results/cifar100_Mobnetv2_torch_training_results_3rd_run.csv"

# Write the header of the CSV file
with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Epoch", "Train Loss", "Train Accuracy", "Train F1-Score", "Test Loss", "Test Accuracy", "Test F1-Score"])

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
f1_train_scores = []
f1_val_scores = []

patience = 10
best_val_loss = float('inf')
epochs_since_improvement = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    all_preds_train = []
    all_targets_train = []

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

            # Store predictions and targets for F1 score calculation
            all_preds_train.extend(predicted.cpu().numpy())
            all_targets_train.extend(targets.cpu().numpy())

            pbar.set_postfix(loss=running_loss / (pbar.n + 1))
            pbar.update()

    avg_train_loss = running_loss / len(train_loader)
    avg_train_accuracy = 100 * correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)

    # Calculate F1 score for training
    f1_train = f1_score(all_targets_train, all_preds_train, average='macro')
    f1_train_scores.append(f1_train)

    print(
        f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.2f}%, F1 Score: {f1_train:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds_val = []
    all_targets_val = []

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc='Validation', unit='batch') as pbar:
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to("cuda")
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # Store predictions and targets for F1 score calculation
                all_preds_val.extend(predicted.cpu().numpy())
                all_targets_val.extend(targets.cpu().numpy())

                pbar.update()
                # Step the scheduler based on the validation loss
                scheduler.step(val_loss)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = 100 * correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)

    # Calculate F1 score for validation
    f1_val = f1_score(all_targets_val, all_preds_val, average='macro')
    f1_val_scores.append(f1_val)

    print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.2f}%, F1 Score: {f1_val:.4f}')

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_since_improvement = 0

    else:
        epochs_since_improvement += 1
        if epochs_since_improvement >= patience and avg_val_loss >= 90:
            print("Early stopping triggered")
            break

    # Save the results to the CSV file
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [epoch + 1, avg_train_loss, avg_train_accuracy, f1_train, avg_val_loss, avg_val_accuracy, f1_val])

    model_save_path = os.path.join(save_dir, f"mobilenet_cifar100v3_torch_epoch{epoch}.pth")
    print(f"Model saved to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

