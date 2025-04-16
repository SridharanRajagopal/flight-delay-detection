import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("preprocessed_flight_data.csv")

# Display first few rows
print("\n🔹 First 5 Rows of the Dataset:")
print(df.head())

# Display dataset info
print("\n🔹 Dataset Info:")
print(df.info())

# Check missing values
print("\n🔹 Missing Values:")
print(df.isnull().sum())

# Fill missing values (Replace NaN with 0 for numerical data)
df.fillna(0, inplace=True)

# Convert categorical columns to category type
df["carrier"] = df["carrier"].astype("category")
df["carrier_name"] = df["carrier_name"].astype("category")
df["airport"] = df["airport"].astype("category")
df["airport_name"] = df["airport_name"].astype("category")

# Display summary statistics
print("\n🔹 Summary Statistics:")
print(df.describe())

# Unique values count for categorical columns
print("\n🔹 Unique Values Count:")
for col in ["carrier", "carrier_name", "airport", "airport_name"]:
    print(f"{col}: {df[col].nunique()} unique values")

# ========================
# 📌 Visualization Section
# ========================

# 1️⃣ Distribution of Flight Delays
plt.figure(figsize=(10, 5))
sns.histplot(df["arr_delay"], bins=50, kde=True)
plt.title("Distribution of Arrival Delays")
plt.xlabel("Arrival Delay (minutes)")
plt.ylabel("Frequency")
plt.show()

# 2️⃣ Top 10 Airports with Most Delays
top_airports = df.groupby("airport_name")["arr_delay"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_airports.values, y=top_airports.index, palette="viridis")
plt.title("Top 10 Airports with Most Delays")
plt.xlabel("Total Delay (minutes)")
plt.ylabel("Airport")
plt.show()

# 3️⃣ Flight Delays by Carrier
plt.figure(figsize=(12, 6))
sns.boxplot(x="carrier", y="arr_delay", data=df)
plt.xticks(rotation=90)
plt.title("Flight Delays by Carrier")
plt.xlabel("Carrier")
plt.ylabel("Arrival Delay (minutes)")
plt.show()

# 4️⃣ Monthly Flight Delay Trend
plt.figure(figsize=(10, 5))
sns.lineplot(x="month", y="arr_delay", data=df, estimator="mean", marker="o")
plt.title("Monthly Flight Delay Trend")
plt.xlabel("Month")
plt.ylabel("Average Arrival Delay (minutes)")
plt.grid(True)
plt.show()

# ========================
# 📌 Correlation Analysis
# ========================

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()

# Save the cleaned dataset
df.to_csv("final_cleaned_flight_data.csv", index=False)

print("\n✅ EDA Completed Successfully & Cleaned Data Saved as 'final_cleaned_flight_data.csv'!")