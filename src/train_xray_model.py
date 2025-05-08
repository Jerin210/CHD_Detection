import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib

# **Load Pre-Split Training Data**
TRAIN_DATASET_PATH = r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\datasets\features\xray_features_train.csv"
TEST_DATASET_PATH = r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\datasets\features\xray_features_test.csv"

df_train = pd.read_csv(TRAIN_DATASET_PATH)
df_test = pd.read_csv(TEST_DATASET_PATH)

# **Ensure the dataset contains the Label column**
if "Label" not in df_train.columns or "Label" not in df_test.columns:
    raise ValueError("The dataset does not contain a 'Label' column!")

# **Extract Features & Labels**
X_train = df_train.drop(columns=["Label"]).values  # Features for training
y_train = df_train["Label"].values  # Labels for training

X_test = df_test.drop(columns=["Label"]).values  # Features for testing
y_test = df_test["Label"].values  # Labels for testing

# **Define XGBoost Classifier**
xgb_model = xgb.XGBClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
)

# **Train XGBoost Model**
xgb_model.fit(X_train, y_train)

# **Generate Predictions**
y_pred = xgb_model.predict(X_test)

# **Evaluate Model**
accuracy = accuracy_score(y_test, y_pred)
print(f"X-ray Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# **Save Trained Model**
MODEL_PATH = r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\modals\xray_xgboost_model.pkl"
joblib.dump(xgb_model, MODEL_PATH)

print(f"X-ray Model Trained & Saved at {MODEL_PATH}")
