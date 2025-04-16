import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
file_path = "cleaned_flight_data.csv"  # Update with actual path
df = pd.read_csv(file_path)

# Handle missing categorical values
df['airport'].fillna(method='ffill', inplace=True)
df['airport_name'].fillna(method='ffill', inplace=True)

# Handle missing numerical values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())  # Fill with median

# Convert arr_del15 to binary target variable (1 = delayed, 0 = not delayed)
df['delayed'] = df['arr_del15'].apply(lambda x: 1 if x > 0 else 0)

# Create new feature: Delay percentage
df['delay_percentage'] = df['arr_del15'] / df['arr_flights']

# Create new feature: Total delay impact
df['total_delay_causes'] = df[['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']].sum(axis=1)

# Encode categorical variables
label_enc = LabelEncoder()
df['carrier'] = label_enc.fit_transform(df['carrier'])
df['airport'] = label_enc.fit_transform(df['airport'])

# Scale numerical columns
scaler = MinMaxScaler()
num_cols = ['arr_flights', 'arr_delay', 'carrier_ct', 'weather_ct', 'nas_ct', 'late_aircraft_ct']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save the preprocessed data
df.to_csv("preprocessed_flight_data.csv", index=False)

print("Preprocessing completed and saved as 'preprocessed_flight_data.csv'.")