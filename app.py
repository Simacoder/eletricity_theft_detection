from fastapi import FastAPI
import pandas as pd
import joblib
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load dataset (Assuming 'data/smart_meter_grid_south_africa.csv' exists)
df = pd.read_csv("data/smart_meter_grid_south_africa.csv")

# Ensure 'Timestamp' column is correctly formatted
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract time-based features
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

# Load trained fraud detection model
model = joblib.load("fraud_detection_model.pkl")  # Ensure this file exists

# Define features used in the model
features = [
    'Hour', 'Day', 'Month', 'DayOfWeek',
    'Energy Consumption (kWh)', 'Voltage (kV)',
    'Frequency (Hz)', 'Power Factor'
]

# Function to detect fraudulent smart meters
def detect_fraud(df):
    df['Fraud_Prediction'] = model.predict(df[features])

    # Filter high-risk (fraudulent) meters
    fraudsters = df[df['Fraud_Prediction'] == 1]

    return fraudsters[['Meter ID', 'Province', 'City', 'Latitude', 'Longitude']].to_dict(orient='records')

# API Endpoint: Return list of fraudulent smart meters
@app.get("/fraudsters")
def get_fraudsters():
    fraud_list = detect_fraud(df)
    return {"fraudsters": fraud_list}

# Run API Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
