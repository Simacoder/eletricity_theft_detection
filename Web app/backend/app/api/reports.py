from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from ..models.reports import Report
from pydantic import BaseModel
from ..crud.reports import create_report, get_reports_by_type, get_all_reports
from ..core.database import get_db

router = APIRouter()

# Pydantic models for request validation and responses
class ReportCreate(BaseModel):
    report_type: str
    content: str
    generated_on: datetime

class ReportResponse(ReportCreate):
    id: int

# Create a new report
@router.post("/", response_model=ReportResponse)
async def create_new_report(report_data: ReportCreate, db: Session = Depends(get_db)):
    # Create a new report in the database
    created_report = create_report(db, report_data)
    
    return created_report

# Get reports by report type
@router.get("/type/{report_type}", response_model=List[ReportResponse])
async def get_reports_by_report_type(report_type: str, db: Session = Depends(get_db)):
    reports = get_reports_by_type(db, report_type)
    if not reports:
        raise HTTPException(status_code=404, detail="No reports found for this type")
    return reports

# Get all reports
@router.get("/", response_model=List[ReportResponse])
async def get_all_reports(db: Session = Depends(get_db)):
    reports = get_all_reports(db)
    return reports
