import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Custom Transform for Resize and Square
class ResizeAndSquare:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        # Resize with padding to maintain aspect ratio
        image = ImageOps.pad(image, (self.size, self.size), color=(0, 0, 0))
        return image

# Transformations are not done yet
train_transform = transforms.Compose([
    ResizeAndSquare(150),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    ResizeAndSquare(150),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load data and annotations
annotations_path = 'annotations.csv'
annotations = pd.read_csv(annotations_path)

image_dir = 'data/images'
image_filenames = os.listdir(image_dir)

# Map id to Material
id_to_material = annotations.set_index('No.')['Material'].to_dict()

# extract ids from image names and map
image_label_mapping = {}
for filename in image_filenames:
    identifier = int(filename.split('_')[0])

    material = id_to_material.get(identifier, None)
    if material:
        image_label_mapping[filename] = material

# print(image_label_mapping)

# Convert labels to binary: 1 if 'plastic', 0 otherwise
binary_image_label_mapping = {
    filename: 1 if material == 'plastic' else 0
    for filename, material in image_label_mapping.items()
}

# print("Binary Image Label Mapping:", binary_image_label_mapping)


class MaterialDataset(Dataset):
    def __init__(self, image_dir, label_mapping, transform=None):
        self.image_dir = image_dir
        self.label_mapping = label_mapping
        self.image_filenames = list(label_mapping.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get image filename and label
        filename = self.image_filenames[idx]
        label = self.label_mapping[filename]

        # Load image
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")  # Convert to RGB

        # Comment if no CLAHE: Apply CLAHE for histogram equalization
        image = self.apply_clahe(image)

        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)

        return image, label

    @staticmethod
    def apply_clahe(image):
        # Convert PIL image to OpenCV format (numpy array)
        image_np = np.array(image)

        # Convert RGB to LAB color space
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

        # Split LAB channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge LAB channels and convert back to RGB
        lab = cv2.merge((l, a, b))
        image_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Convert back to PIL Image
        return Image.fromarray(image_np)


__all__ = ["MaterialDataset", "binary_image_label_mapping", "image_dir"]
