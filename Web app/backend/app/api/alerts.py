from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List
from ..models.alerts import Alert
from ..crud.alerts import create_alert, get_alerts_by_meter_id, get_all_alerts
from ..core.database import get_db

router = APIRouter()

# Pydantic models for creating and responding with alerts
class AlertCreate(BaseModel):
    meter_id: int
    anomaly_id: int
    alert_message: str
    alert_severity: str

class AlertResponse(AlertCreate):
    id: int
    alert_timestamp: str

# Create a new alert
@router.post("/", response_model=AlertResponse)
async def create_new_alert(alert_data: AlertCreate, db: AsyncSession = Depends(get_db)):
    created_alert = await create_alert(db, alert_data)
    return created_alert

# Get alerts for a specific meter
@router.get("/api/meter/{meter_id}", response_model=List[AlertResponse])
async def get_alerts_for_meter(meter_id: int, db: AsyncSession = Depends(get_db)):
    alerts = await get_alerts_by_meter_id(db, meter_id)
    if not alerts:
        raise HTTPException(status_code=404, detail="No alerts found for this meter")
    return alerts

# Get all alerts
@router.get("/", response_model=List[AlertResponse])
async def get_all_alerts(db: AsyncSession = Depends(get_db), limit: int = 100, offset: int = 0):
    if limit <= 0:
        return []
    
    alerts = await db.fetch_alerts(limit=limit, offset=offset)
    
    if alerts:
        return alerts + await get_all_alerts(db, limit=limit-10, offset=offset+10)
    else:
        return []

