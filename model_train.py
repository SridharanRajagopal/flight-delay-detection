import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Load cleaned data
data = pd.read_csv("data/cleaned_flight_data (1).csv")

# Encode categorical features
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Split features and target
X = data.drop(columns=['delay_binary'])
y = data['delay_binary']

# 🔍 Print the actual feature column list for reference
print("🧠 Features used for training:")
print(X.columns.tolist())

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, rf_model.predict(X_train))
test_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"\n📊 Train Accuracy: {train_acc:.4f}")
print(f"📊 Test Accuracy: {test_acc:.4f}")

# Save model and scaler
os.makedirs("models", exist_ok=True)
with open("models/flight_delay_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model and Scaler saved successfully!")
