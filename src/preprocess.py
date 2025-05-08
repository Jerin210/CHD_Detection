import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# Define Augmentation Pipeline
augmentor = A.Compose([
    A.Rotate(limit=10, p=0.5),  # Rotate within ±10 degrees
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.HorizontalFlip(p=0.5)  # Apply 50% chance of flipping
])

# Function to Apply Augmentation & Save Images
def preprocess_and_save(images, save_dir, augment=True):
    os.makedirs(save_dir, exist_ok=True)
    for idx, img_path in tqdm(enumerate(images), total=len(images)):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
        img = cv2.resize(img, (224, 224))  # Resize to match CNN input

        # Save Original Image
        cv2.imwrite(os.path.join(save_dir, f"{idx}.png"), img)

        if augment:
            aug_img = augmentor(image=img)["image"]
            cv2.imwrite(os.path.join(save_dir, f"{idx}_aug.png"), aug_img)

# Define Directories
RAW_DIR = r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\datasets\xray_images\raw"
TRAIN_DIR = r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\datasets\xray_images\processed\train"
TEST_DIR = r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\datasets\xray_images\processed\test"

# Get Image Paths
chd_images = [os.path.join(RAW_DIR, "CHD", img) for img in os.listdir(os.path.join(RAW_DIR, "CHD"))]
no_chd_images = [os.path.join(RAW_DIR, "No_CHD", img) for img in os.listdir(os.path.join(RAW_DIR, "No_CHD"))]

# Split into Train & Test (80-20)
from sklearn.model_selection import train_test_split
train_chd, test_chd = train_test_split(chd_images, test_size=0.2, random_state=42)
train_no_chd, test_no_chd = train_test_split(no_chd_images, test_size=0.2, random_state=42)

# Apply Preprocessing & Augmentation
preprocess_and_save(train_chd, os.path.join(TRAIN_DIR, "CHD"), augment=True)
preprocess_and_save(train_no_chd, os.path.join(TRAIN_DIR, "No_CHD"), augment=True)
preprocess_and_save(test_chd, os.path.join(TEST_DIR, "CHD"), augment=False)  # No augmentation on test
preprocess_and_save(test_no_chd, os.path.join(TEST_DIR, "No_CHD"), augment=False)

print("Dataset Preprocessing & Augmentation Completed!")
