from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
from ..model_loader import load_model

model = load_model()
if model is None:
    raise Exception("Model failed to load.")

class MeterDataRequest(BaseModel):
    meter_id: int
    daily_consumption: float
    base_consumption: float
    other_features: Dict[str, float]

router = APIRouter()

@router.post("/api/predict_fraud")
async def predict_fraud(data: MeterDataRequest):
    try:
        input_features = [
            data.daily_consumption,
            data.base_consumption,
            *data.other_features.values(),
        ]
        
        prediction = model.predict([input_features])

        return {"meter_id": data.meter_id, "fraud_prediction": int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error predicting fraud: {e}")

