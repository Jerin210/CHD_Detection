import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# 🔹 Define Paths
# -----------------------------
DATASET_PATH = r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\datasets\xray_images\balanced"
CSV_SAVE_PATH = r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\datasets\xray_features.csv"

IMG_SIZE = (224, 224)

# -----------------------------
# 🔹 Load Pretrained Model (Feature Extractor)
# -----------------------------
base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")

# -----------------------------
# 🔹 Function to Extract Features
# -----------------------------
def extract_features(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Normalize
    features = base_model.predict(img_array)
    return features.flatten()  # Convert to 1D vector

# -----------------------------
# 🔹 Process X-ray Images
# -----------------------------
data = []
labels = []

for label in ["CHD", "No_CHD"]:
    folder_path = os.path.join(DATASET_PATH, "train", label)
    
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        feature_vector = extract_features(img_path)  # Extract CNN features
        data.append(feature_vector)
        labels.append(label)

# Convert to DataFrame
df = pd.DataFrame(data)
df["Label"] = labels

# Save to CSV
df.to_csv(CSV_SAVE_PATH, index=False)
print(f"X-ray features saved at {CSV_SAVE_PATH}")
