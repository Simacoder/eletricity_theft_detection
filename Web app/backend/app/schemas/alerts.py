from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class AlertCreate(BaseModel):
    anomaly_id: int
    alert_timestamp: datetime
    alert_message: str
    severity: str

    class Config:
        from_attributes = True

class AlertUpdate(BaseModel):
    alert_message: Optional[str] = None
    severity: Optional[str] = None 

    class Config:
        from_attributes = True

class AlertResponse(AlertCreate):
    id: int
    alert_timestamp: datetime

    class Config:
        from_attributes = True
