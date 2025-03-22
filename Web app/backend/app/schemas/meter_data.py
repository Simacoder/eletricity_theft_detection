from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class MeterDataCreate(BaseModel):
    meter_id: int
    reading_time: datetime
    value: float
    consumption_pattern: str

    class Config:
        from_attributes = True

class MeterDataUpdate(BaseModel):
    reading_time: Optional[datetime] = None
    value: Optional[float] = None
    consumption_pattern: Optional[str] = None

    class Config:
        from_attributes = True

class MeterDataResponse(MeterDataCreate):
    id: int

    class Config:
        from_attributes = True
