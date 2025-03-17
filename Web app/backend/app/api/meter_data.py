from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from ..models.meter_data import MeterData
from ..crud.meter_data import create_meter_data, get_meter_data_by_id, get_all_meter_data
from ..core.database import get_db
import joblib
import numpy as np

router = APIRouter()

model = joblib.load("models/fraud_detection_model.pkl")

# Pydantic model for request validation
class MeterDataCreate(BaseModel):
    meter_id: int
    timestamp: str
    value: float

class MeterDataResponse(MeterDataCreate):
    id: int

# Create new meter data entry
@router.post("/", response_model=MeterDataResponse)
async def create_meter(meter_data: MeterDataCreate, db: Session = Depends(get_db)):
    # Create a new meter data entry in the database
    created_meter_data = create_meter_data(db, meter_data)
    
    # Flag any anomalies using the fraud detection model
    flag_anomaly(meter_data)
    
    return created_meter_data

# Get meter data by ID
@router.get("/{meter_id}", response_model=List[MeterDataResponse])
async def get_meter_by_id(meter_id: int, db: Session = Depends(get_db)):
    meter_data = get_meter_data_by_id(db, meter_id)
    if not meter_data:
        raise HTTPException(status_code=404, detail="Meter data not found")
    return meter_data

# Get all meter data
@router.get("/", response_model=List[MeterDataResponse])
async def get_all_meter(db: Session = Depends(get_db)):
    all_meter_data = get_all_meter_data(db)
    return all_meter_data

# Function to flag anomalies in the meter data using the fraud detection model
def flag_anomaly(meter_data: MeterDataCreate):
    data = np.array([[meter_data.value]])
    prediction = model.predict(data)

    if prediction == 1:
        # Flag as anomalous if prediction is 1
        print(f"Anomaly detected for Meter ID {meter_data.meter_id} at timestamp {meter_data.timestamp}")
    else:
        print(f"Meter ID {meter_data.meter_id} is normal at timestamp {meter_data.timestamp}")
