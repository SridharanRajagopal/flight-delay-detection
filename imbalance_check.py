import pandas as pd

# Load dataset
df = pd.read_csv("D:/Capstone_Project/data/final_cleaned_flight_data (1).csv")

# Define the target column
target_col = "arr_del15"

# Check if target column exists
if target_col in df.columns:
    # Count occurrences of each class
    class_counts = df[target_col].value_counts()

    # Calculate imbalance ratio
    imbalance_ratio = class_counts.min() / class_counts.max()

    print("/n🔹 **Class Distribution:**")
    print(class_counts)

    print("/n🔹 **Class Imbalance Ratio:** {:.4f}".format(imbalance_ratio))

    # Interpretation
    if imbalance_ratio < 0.5:
        print("/n⚠️ **Severe Imbalance Detected!** Consider resampling techniques.")
    elif imbalance_ratio < 0.8:
        print("/n⚠️ **Moderate Imbalance Detected!** Resampling might be needed.")
    else:
        print("/n✅ **Dataset is balanced! No resampling needed.**")

else:
    print(f"❌ Target column '{target_col}' not found.")