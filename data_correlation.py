import pandas as pd
import numpy as np

# Load the preprocessed dataset
file_path = "preprocessed_flight_data.csv"  # Update this if needed
df = pd.read_csv(file_path)

# Display the first few rows
print("\nFirst 5 Rows:\n", df.head())

# Show dataset info
print("\nDataset Info:\n")
print(df.info())

# Summary statistics for numerical columns
print("\nSummary Statistics:\n")
print(df.describe())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Unique values in categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\nUnique values in '{col}': {df[col].nunique()}")

# Count of delayed vs non-delayed flights (if 'delayed' column exists)
if 'delayed' in df.columns:
    print("\nFlight Delay Distribution:\n", df['delayed'].value_counts())

# Correlation matrix for numerical columns
numeric_df = df.select_dtypes(include=['number'])
print("\nFeature Correlations:\n")
print(numeric_df.corr())

# Save the correlation matrix as a CSV file (optional)
numeric_df.corr().to_csv("correlation_matrix.csv")
print("\nCorrelation matrix saved as 'correlation_matrix.csv'")