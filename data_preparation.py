import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import cv2
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

# Custom Transform for Resize and Square
class ResizeAndSquare:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        # Resize with padding to maintain aspect ratio
        image = ImageOps.pad(image, (self.size, self.size), color=(0, 0, 0))
        return image

# Transformations
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
annotations_path = 'annotations_final.csv'
annotations = pd.read_csv(annotations_path)

image_dir = 'i190_data'
synthetic_image_dir = 'data/synthetic_images' # saving images that will be created with SMOTE
os.makedirs(synthetic_image_dir, exist_ok=True)
image_filenames = os.listdir(image_dir)

# Map id to Material
id_to_material = annotations.set_index('No.')['Material'].to_dict()

# extract ids from image names and map
image_label_mapping = {}
for filename in image_filenames:
    base_identifier = int(filename.split('_')[0])  # Extract the numeric identifier
    material = id_to_material.get(base_identifier, None)  # Lookup using base identifier
    if material:
        image_label_mapping[filename] = material
# print(image_label_mapping)

# Convert labels to binary: 1 if 'plastic', 0 otherwise
binary_image_label_mapping = {
    filename: 1 if material == 'plastic' else 0
    for filename, material in image_label_mapping.items()
}

# print("Binary Image Label Mapping:", binary_image_label_mapping)



# Prepare features for SMOTE
def prepare_features_and_labels(image_dir, label_mapping):
    features = []
    labels = []
    for filename, label in label_mapping.items():
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("RGB")  # Convert to RGB
        image = image.resize((150, 150))  # Match transform size
        features.append(np.array(image).flatten())  # Flatten image
        labels.append(label)
    return np.array(features), np.array(labels)

# Extract features and labels
features, labels = prepare_features_and_labels(image_dir, binary_image_label_mapping)

# Apply SMOTE to balance the dataset
print("Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
augmented_features, augmented_labels = smote.fit_resample(features, labels)

print(f"Original dataset size: {len(labels)}")
print(f"Augmented dataset size: {len(augmented_labels)}")

# Save synthetic images
for i, feature in enumerate(augmented_features[len(labels):]):
    image = feature.reshape(150, 150, 3).astype(np.uint8)
    synthetic_filename = f"aug_{i}.jpg"
    synthetic_path = os.path.join(synthetic_image_dir, synthetic_filename)
    Image.fromarray(image).save(synthetic_path)

# Update binary_image_label_mapping to include synthetic images
for i, label in enumerate(augmented_labels[len(labels):]):
    binary_image_label_mapping[f"aug_{i}.jpg"] = label  # Use only filename


# Combine real and synthetic directories
full_image_dir = [image_dir, synthetic_image_dir]

# Check class distribution after SMOTE
class_counts = Counter(augmented_labels)
total_images = len(augmented_labels)
class_proportions = {cls: count / total_images for cls, count in class_counts.items()}

print("Class Distribution After SMOTE:")
for cls, count in class_counts.items():
    print(f"Class {cls}: {count} images ({class_proportions[cls]*100:.2f}%)")


class MaterialDataset(Dataset):
    def __init__(self, image_dirs, label_mapping, transform=None):
        self.image_dirs = image_dirs  # List of directories (real + synthetic)
        self.label_mapping = label_mapping
        self.image_filenames = list(label_mapping.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get image filename and label
        filename = self.image_filenames[idx]
        label = self.label_mapping[filename]

        # Determine directory: check if file is synthetic
        if filename.startswith("aug_"):
            image_dir = self.image_dirs[1]  # Synthetic images directory
        else:
            image_dir = self.image_dirs[0]  # Real images directory

        # Load image
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("RGB")  # Convert to RGB

        # Apply CLAHE
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



_all_ = ["MaterialDataset", "binary_image_label_mapping", "image_dir", "synthetic_image_dir"]