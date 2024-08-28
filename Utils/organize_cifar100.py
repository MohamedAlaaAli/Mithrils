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

# CIFAR-100 classes
classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Create directories for CIFAR-100
for phase in ['train', 'test']:
    for cls in classes:
        os.makedirs(f'./data/CIFAR100/{phase}/{cls}', exist_ok=True)

# Function to save images
def save_images(dataset, phase):
    for i, (img, label) in enumerate(tqdm(dataset)):
        class_name = classes[label]
        img.save(f'./data/CIFAR100/{phase}/{class_name}/{i:04d}.png')

# Load CIFAR-100 data
train_data = datasets.CIFAR100(root='./data', train=True, download=True)
test_data = datasets.CIFAR100(root='./data', train=False, download=True)

# Save training images
save_images(train_data, 'train')

# Save testing images
save_images(test_data, 'test')
