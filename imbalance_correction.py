import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Raw Dataset
data = pd.read_csv("D:/Capstone_Project/data/final_cleaned_flight_data (1).csv")

# Handle Missing Values for numeric features only
# Select only numeric columns
numeric_data = data.select_dtypes(include=np.number)

# Fill missing values in numeric columns with their median
numeric_data.fillna(numeric_data.median(), inplace=True)

# Replace numeric columns in the original DataFrame with filled values
data[numeric_data.columns] = numeric_data

# Fix Target Variable: Convert `arr_del15` to Binary (0 = No Delay, 1 = Delayed)
data['delay_binary'] = (data['arr_del15'] > 15).astype(int)

# Drop original `arr_del15` since it's now transformed
data.drop(columns=['arr_del15'], inplace=True)

# Encode Categorical Features (Carrier & Airport)
le_carrier = LabelEncoder()
le_airport = LabelEncoder()
data['carrier'] = le_carrier.fit_transform(data['carrier'])
data['airport'] = le_airport.fit_transform(data['airport'])

# Standardization of Numeric Features
scaler = StandardScaler()
numeric_features = ['arr_flights', 'carrier_ct', 'weather_ct', 'nas_ct',
                    'security_ct', 'late_aircraft_ct', 'arr_cancelled', 'arr_diverted']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Check Class Imbalance
class_counts = data['delay_binary'].value_counts()
class_percentage = (class_counts / len(data)) * 100

print("Class Distribution:/n", class_counts)
print("/nClass Percentage:/n", class_percentage)

# Visualize Class Imbalance
plt.figure(figsize=(6, 4))
class_counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.xticks(ticks=[0, 1], labels=['Not Delayed (0)', 'Delayed (1)'], rotation=0)
plt.xlabel("Flight Delay Status")
plt.ylabel("Number of Flights")
plt.title("Class Imbalance in Flight Delay Data")
plt.show()

# Save Cleaned Data
data.to_csv("cleaned_flight_data.csv", index=False)

print("/n✅ Data cleaning and processing completed. Cleaned file saved as 'cleaned_flight_data.csv'.")