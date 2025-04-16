import pandas as pd

# Load the dataset
file_path = "D:/Capstone_Project/data/raw.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Display basic info
print("Initial Data Info:/n", df.info())

# Check for missing values
print("Missing Values:/n", df.isnull().sum())

# Fill missing values (if any)
df.fillna(0, inplace=True)  # Replace missing values with 0, modify if needed

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert data types if necessary (e.g., year and month as integers)
df['year'] = df['year'].astype(int)
df['month'] = df['month'].astype(int)

# Check for outliers (optional: use boxplots or z-score methods)
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_columns].describe()

# Save the cleaned dataset
cleaned_file_path = "cleaned_flight_data.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to {cleaned_file_path}")