import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, real=True, fourier=False, transform=None):
        self.data_dir = data_dir
        self.classes = os.listdir(data_dir)
        self.class_to_index = {class_: index for index, class_ in enumerate(self.classes)}
        self.transform = transform
        self.img_paths = self.get_img_paths()
        self.real = real
        self.fourier = fourier

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        if self.real and not self.fourier:
            return self.get_rgb_images(index)
        elif self.fourier and not self.real:
            return self.get_fft_images(index)
        elif self.real and self.fourier:
            return self.get_fusion_images(index)
        else:
            raise ValueError("Either 'real' or 'fourier' must be True.")

    def get_rgb_images(self, index):
        img_path, label = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_fft_images(self, index):
        img_path, label = self.img_paths[index]
        img = Image.open(img_path).convert("L")
        img_fft_mag = self.compute_fft(img.to("cuda" if torch.cuda.is_available() else "cpu"))
        return img_fft_mag, label

    def get_fusion_images(self, index):
        img_path, label = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        if self.transform:
            img = self.transform(img)
        img_fft_mag = self.compute_fft(img_gray)
        img_fft_mag_3channel = img_fft_mag.repeat(3, 1, 1)  # Assuming the FFT magnitude is single-channel
        fused_img = torch.cat((img, img_fft_mag_3channel), dim=0)  # Concatenating along the channel dimension
        return fused_img, label

    def compute_fft(self, img):
        img_tensor = transforms.ToTensor()(img)
        img_fft = torch.fft.fft2(img_tensor)
        img_fft_shifted = torch.fft.fftshift(img_fft)
        img_mag = torch.log(torch.abs(img_fft_shifted) + 1e-8)
        img_mag_norm = (img_mag - torch.min(img_mag)) / (torch.max(img_mag) - torch.min(img_mag))
        return img_mag_norm

    def get_img_paths(self):
        paths = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for img in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img)
                label = self.class_to_index[class_name]
                paths.append((img_path, label))
        return paths
