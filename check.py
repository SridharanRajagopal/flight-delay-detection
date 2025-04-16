import pickle
import pandas as pd

# Paths to files
model_path = 'D:/Capstone_Project/models/flight_delay_model.pkl'
scaler_path = 'D:/Capstone_Project/models/scaler.pkl'

# Load and inspect the model
def inspect_model():
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            print("Model Loaded Successfully!")
            print(f"Model Type: {type(model)}")
            if hasattr(model, 'feature_names_in_'):
                print("Model Features:", model.feature_names_in_)
            else:
                print("Model does not store feature names directly.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Load and inspect the scaler
def inspect_scaler():
    try:
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
            print("Scaler Loaded Successfully!")
            print(f"Scaler Type: {type(scaler)}")
            if hasattr(scaler, 'feature_names_in_'):
                print("Scaler Features:", scaler.feature_names_in_)
            else:
                print("Scaler does not store feature names directly.")
    except Exception as e:
        print(f"Error loading scaler: {e}")

# Run inspections
if __name__ == '__main__':
    inspect_model()
    print("-"*50)
    inspect_scaler()
