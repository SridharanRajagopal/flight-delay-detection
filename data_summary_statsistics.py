import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("D:/Capstone_Project/data/final_cleaned_flight_data (1).csv")

# Display first few rows
print("/n🔹 First 5 Rows of the Dataset:")
print(df.head())

# Display dataset info
print("/n🔹 Dataset Info:")
print(df.info())

# Check missing values
print("/n🔹 Missing Values:")
print(df.isnull().sum())

# Summary statistics of numerical columns
print("/n🔹 Summary Statistics:")
print(df.describe())

# Unique values count for categorical columns
print("/n🔹 Unique Values Count:")
for col in df.select_dtypes(include=["object", "category"]).columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Compute correlation for numeric columns only
numeric_df = df.select_dtypes(include=["number"])  # Select only numeric columns
print("/n🔹 Feature Correlations:")
print(numeric_df.corr())