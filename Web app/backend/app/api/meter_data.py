from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from ..models.meter_data import MeterData
from ..crud.meter_data import create_meter_data, get_meter_data_by_id, get_all_meter_data
from ..core.database import get_db
import joblib
import numpy as np
import os

router = APIRouter()

# Path to the fraud detection model
model_path = "backend/app/models/fraud_detection_model.pkl"

# Load the model only once when the application starts
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

# Pydantic model for creating meter data entries
class MeterDataCreate(BaseModel):
    meter_id: int
    timestamp: str
    value: float

# Pydantic model for the response, including the ID of the meter data entry
class MeterDataResponse(MeterDataCreate):
    id: int

# Create new meter data entry
@router.post("/", response_model=MeterDataResponse)
async def create_meter(meter_data: MeterDataCreate, db: Session = Depends(get_db)):
    try:
        # Create a new meter data entry in the database
        created_meter_data = create_meter_data(db, meter_data)
        
        # Flag any anomalies using the fraud detection model
        flag_anomaly(meter_data)
        
        return created_meter_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating meter data: {e}")

# Get meter data by meter ID
@router.get("/{meter_id}", response_model=List[MeterDataResponse])
async def get_meter_by_id(meter_id: int, db: Session = Depends(get_db)):
    try:
        meter_data = get_meter_data_by_id(db, meter_id)
        if not meter_data:
            raise HTTPException(status_code=404, detail="Meter data not found")
        return meter_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching meter data: {e}")

# Get all meter data
@router.get("/", response_model=List[MeterDataResponse])
async def get_all_meter(db: Session = Depends(get_db)):
    try:
        all_meter_data = get_all_meter_data(db)
        return all_meter_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching all meter data: {e}")

# Function to flag anomalies in the meter data using the fraud detection model
def flag_anomaly(meter_data: MeterDataCreate):
    try:
        data = np.array([[meter_data.value]])
        
        prediction = model.predict(data)
        
        if prediction == 1:
            print(f"Anomaly detected for Meter ID {meter_data.meter_id} at timestamp {meter_data.timestamp}")
        else:
            print(f"Meter ID {meter_data.meter_id} is normal at timestamp {meter_data.timestamp}")
    
    except Exception as e:
        print(f"Error in anomaly detection for Meter ID {meter_data.meter_id}: {e}")
