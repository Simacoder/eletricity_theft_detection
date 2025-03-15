from sqlalchemy.orm import Session
from ..models import reports as reports_model
from ..schemas import reports as reports_schema
from datetime import datetime

def create_report(db: Session, report: reports_schema.ReportCreate):
    db_report = reports_model.Report(
        report_type=report.report_type,
        report_data=report.report_data,
        created_at=datetime.now(),
        generated_by=report.generated_by
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report

def get_reports(db: Session, skip: int = 0, limit: int = 100):
    return db.query(reports_model.Report).offset(skip).limit(limit).all()

def get_report_by_id(db: Session, report_id: int):
    return db.query(reports_model.Report).filter(reports_model.Report.report_id == report_id).first()

def get_reports_by_type(db: Session, report_type: str):
    return db.query(reports_model.Report).filter(reports_model.Report.report_type == report_type).all()

def update_report(db: Session, report_id: int, report: reports_schema.ReportUpdate):
    db_report = db.query(reports_model.Report).filter(reports_model.Report.report_id == report_id).first()
    
    if db_report:
        db_report.report_data = report.report_data if report.report_data else db_report.report_data
        db_report.report_type = report.report_type if report.report_type else db_report.report_type
        db.commit()
        db.refresh(db_report)
    
    return db_report

def delete_report(db: Session, report_id: int):
    db_report = db.query(reports_model.Report).filter(reports_model.Report.report_id == report_id).first()
    if db_report:
        db.delete(db_report)
        db.commit()
    return db_report
