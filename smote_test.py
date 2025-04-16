import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Cleaned Data
data = pd.read_csv("D:/Capstone_Project/data/cleaned_flight_data (1).csv")

# Identify Categorical Columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Encode Categorical Features
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Split Features and Target
X = data.drop(columns=['delay_binary'])  # Features
y = data['delay_binary']  # Target Variable

# Standardize Numerical Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and Train Logistic Regression Model
log_reg = LogisticRegression(solver='liblinear', random_state=42)
log_reg.fit(X_train, y_train)

# Predictions on Training and Test Set
y_train_pred = log_reg.predict(X_train)
y_test_pred = log_reg.predict(X_test)

# Evaluate Training Performance
train_accuracy = accuracy_score(y_train, y_train_pred)
train_report = classification_report(y_train, y_train_pred)

# Evaluate Test Performance
test_accuracy = accuracy_score(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred)

# Display Results
print(f"/n📊 Training Accuracy: {train_accuracy:.4f}")
print(f"/n📊 Test Accuracy: {test_accuracy:.4f}")

print("/n📌 Training Set Classification Report:/n", train_report)
print("/n📌 Test Set Classification Report:/n", test_report)

# Check Overfitting
if train_accuracy - test_accuracy > 0.05:
    print("/n⚠️ Possible Overfitting Detected! Training accuracy is significantly higher than test accuracy.")
else:
    print("/n✅ No significant overfitting detected. Model generalizes well.")
