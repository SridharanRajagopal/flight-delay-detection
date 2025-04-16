import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Cleaned Data
data = pd.read_csv("D:/Capstone_Project/data/cleaned_flight_data (1).csv")

# Identify Categorical Columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Encode Categorical Features Using Label Encoding
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

# Make Predictions
y_pred = log_reg.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print Results
print(f"/n✅ Logistic Regression Model Accuracy: {accuracy:.4f}/n")
print("📊 Confusion Matrix:/n", conf_matrix)
print("/n📌 Classification Report:/n", class_report)