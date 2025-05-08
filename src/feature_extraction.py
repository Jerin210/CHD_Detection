import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Load Pretrained ResNet50 Model (Without Classification Head)
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Define Image Directories
TRAIN_DIR = r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\datasets\xray_images\processed\train"
TEST_DIR = r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\datasets\xray_images\processed\test"

# Function to Extract Features
def extract_features(img_dir):
    features, labels = [], []
    
    for label in ["CHD", "No_CHD"]:
        class_dir = os.path.join(img_dir, label)
        for img_name in tqdm(os.listdir(class_dir), desc=f"Processing {label}"):
            img_path = os.path.join(class_dir, img_name)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)  # Normalize for ResNet50

            # Extract Features
            feature = base_model.predict(img_array)
            features.append(feature.flatten())
            labels.append(1 if label == "CHD" else 0)

    return np.array(features), np.array(labels)

# Extract Features for Train & Test Sets
X_train, y_train = extract_features(TRAIN_DIR)
X_test, y_test = extract_features(TEST_DIR)

# Save as CSV
train_df = pd.DataFrame(X_train)
train_df["Label"] = y_train
train_df.to_csv(r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\datasets\features\xray_features_train.csv", index=False)

test_df = pd.DataFrame(X_test)
test_df["Label"] = y_test
test_df.to_csv(r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\datasets\features\xray_features_test.csv", index=False)

print("Feature extraction completed and saved as CSV!")
