import os
from torchvision import datasets
from tqdm import tqdm

# MNIST classes
classes = [str(i) for i in range(10)]

# Create directories
for phase in ['train', 'test']:
    for cls in classes:
        os.makedirs(f'./data/MNIST/{phase}/{cls}', exist_ok=True)

# Function to save images
def save_images(dataset, phase):
    for i, (img, label) in enumerate(tqdm(dataset)):
        class_name = classes[label]
        img.save(f'./data/MNIST/{phase}/{class_name}/{i:04d}.png')

# Load MNIST data
train_data = datasets.MNIST(root='./data', train=True, download=True)
test_data = datasets.MNIST(root='./data', train=False, download=True)

# Save training images
save_images(train_data, 'train')

# Save testing images
save_images(test_data, 'test')
