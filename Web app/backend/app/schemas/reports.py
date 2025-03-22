from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ReportCreate(BaseModel):
    report_type: str
    report_data: str
    generated_by: str

    class Config:
        from_attributes = True

class ReportUpdate(BaseModel):
    report_type: Optional[str] = None
    report_data: Optional[str] = None
    generated_by: Optional[str] = None

    class Config:
        from_attributes = True

class ReportResponse(ReportCreate):
    report_id: int
    created_at: datetime

    class Config:
        from_attributes = True
