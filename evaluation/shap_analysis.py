import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import shap  # For Explainability

# Paths
X_TEST_PATH = r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\datasets\features\xray_features_test.csv"
MODEL_PATH = r"C:\Users\sebas\OneDrive\Desktop\Major_Project\CHD_D\CHD_Detection_project\modals\xray_xgboost_model.pkl"

# Load Test Dataset
df_test = pd.read_csv(X_TEST_PATH)

# Feature Columns & Target Column
X_test = df_test.iloc[:, :-1].values  # Features
y_test = df_test.iloc[:, -1].values   # Labels

# Convert Labels if String
if isinstance(y_test[0], str):
    y_test = np.where(y_test == "CHD", 1, 0)

# Load Trained Model
model = joblib.load(MODEL_PATH)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Accuracy & Report
print(f"\nX-ray Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No CHD", "CHD"], yticklabels=["No CHD", "CHD"])
plt.xlabel("Detected")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Feature Importance
plt.figure(figsize=(8, 5))
xgb.plot_importance(model, max_num_features=10, importance_type="gain", title="Top 10 Feature Importances")
plt.show()

print("\nGenerating SHAP Analysis...")

# Create SHAP Explainer
explainer = shap.Explainer(model, X_test)

# Calculate SHAP Values
shap_values = explainer(X_test)

# Summary Plot (Feature Impact)
shap.summary_plot(shap_values, X_test, show=True, plot_type="bar")

# Detailed Beeswarm Plot
shap.summary_plot(shap_values, X_test, show=True)

# Force Plot (Optional for individual prediction)
# shap.initjs()
# shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])

print("SHAP Analysis Completed.")
