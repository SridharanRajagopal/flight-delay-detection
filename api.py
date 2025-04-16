import uvicorn
import pickle
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load("flight_delay_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define FastAPI app
app = FastAPI()

# Define the request payload structure
class FlightInput(BaseModel):
    year: int
    month: int
    carrier: int
    airport: int
    arr_flights: int
    arr_del15: int
    carrier_ct: float
    weather_ct: float
    nas_ct: float
    security_ct: float
    late_aircraft_ct: float
    arr_cancelled: int
    arr_diverted: int
    arr_delay: int
    carrier_delay: int
    weather_delay: int
    nas_delay: int
    security_delay: int
    late_aircraft_delay: int

# Prediction endpoint
@app.post("/predict")
def predict_delay(data: FlightInput):
    # Convert input to a DataFrame
    input_data = pd.DataFrame([data.dict().values()], columns=data.dict().keys())

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Return result
    return {"delay_prediction": int(prediction), "probability": float(probability)}

# Fix asyncio issue in Jupyter Notebook
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Run API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
