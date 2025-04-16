import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# Initialize and Train XGBoost Model
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6
)

xgb_model.fit(X_train, y_train)

# Predictions
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# Evaluate Training Performance
train_accuracy = accuracy_score(y_train, y_train_pred)
train_report = classification_report(y_train, y_train_pred)

# Evaluate Test Performance
test_accuracy = accuracy_score(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred)

# Display Results
print(f"/n📊 XGBoost Training Accuracy: {train_accuracy:.4f}")
print(f"/n📊 XGBoost Test Accuracy: {test_accuracy:.4f}")

print("/n📌 Training Set Classification Report:/n", train_report)
print("/n📌 Test Set Classification Report:/n", test_report)

# Check Overfitting
if train_accuracy - test_accuracy > 0.05:
    print("/n⚠️ Possible Overfitting Detected! Training accuracy is significantly higher than test accuracy.")
else:
    print("/n✅ No significant overfitting detected. Model generalizes well.")

# Feature Importance
xgb.plot_importance(xgb_model, max_num_features=10)
plt.title("Top 10 Important Features in XGBoost Model")
plt.show()