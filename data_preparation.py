import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


# Transformations are not done yet
train_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((150, 150)),
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

        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)

        return image, label



__all__ = ["MaterialDataset", "binary_image_label_mapping", "image_dir"]