from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from datetime import datetime
from ..models.reports import Report
from pydantic import BaseModel
from ..crud.reports import create_report, get_reports_by_type, get_all_reports
from ..core.database import get_db

router = APIRouter()

class ReportCreate(BaseModel):
    report_type: str
    content: str
    generated_on: datetime

class ReportResponse(ReportCreate):
    id: int

@router.post("/", response_model=ReportResponse)
async def create_new_report(report_data: ReportCreate, db: AsyncSession = Depends(get_db)):
    created_report = await create_report(db, report_data)
    return created_report

@router.get("/api/type/{report_type}", response_model=List[ReportResponse])
async def get_reports_by_report_type(report_type: str, db: AsyncSession = Depends(get_db)):
    reports = await get_reports_by_type(db, report_type)
    if not reports:
        raise HTTPException(status_code=404, detail="No reports found for this type")
    return reports

@router.get("/", response_model=List[ReportResponse])
async def get_all_reports(db: AsyncSession = Depends(get_db)):
    reports = await get_all_reports(db)
    return reports
