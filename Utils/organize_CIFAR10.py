import os
import random
import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm

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
seed = 42
set_seed(seed)

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create directories
for phase in ['train', 'test']:
    for cls in classes:
        os.makedirs(f'./data/CIFAR10/{phase}/{cls}', exist_ok=True)

# Function to save images
def save_images(dataset, phase):
    for i, (img, label) in enumerate(tqdm(dataset)):
        class_name = classes[label]
        img.save(f'./data/CIFAR10/{phase}/{class_name}/{i:04d}.png')

# Load CIFAR-10 data
train_data = datasets.CIFAR10(root='./data', train=True, download=True)
test_data = datasets.CIFAR10(root='./data', train=False, download=True)

# Save training images
save_images(train_data, 'train')

# Save testing images
save_images(test_data, 'test')

