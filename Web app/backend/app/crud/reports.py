from sqlalchemy.future import select
from ..models import reports as reports_model
from ..schemas import reports as reports_schema
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os

database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise ValueError("DATABASE_URL environment variable is not set")

async_engine = create_async_engine(database_url, echo=True, future=True)

async_session = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def create_report(db: AsyncSession, report: reports_schema.ReportCreate):
    try:
        db_report = reports_model.Report(
            report_type=report.report_type,
            report_data=report.report_data,
            created_at=datetime.utcnow(),
            generated_by=report.generated_by
        )
        
        async with db.begin():
            db.add(db_report)
        
        await db.commit()
        await db.refresh(db_report)
        return db_report
    except Exception as e:
        await db.rollback()
        raise Exception(f"Error creating report: {e}")

async def get_reports(db: AsyncSession, skip: int = 0, limit: int = 100):
    try:
        result = await db.execute(select(reports_model.Report).offset(skip).limit(limit))
        return result.scalars().all()
    except Exception as e:
        raise Exception(f"Error retrieving reports: {e}")

async def get_report_by_id(db: AsyncSession, report_id: int):
    try:
        result = await db.execute(select(reports_model.Report).filter(reports_model.Report.report_id == report_id))
        return result.scalar_one_or_none()
    except Exception as e:
        raise Exception(f"Error retrieving report by ID {report_id}: {e}")

async def get_reports_by_type(db: AsyncSession, report_type: str):
    try:
        result = await db.execute(select(reports_model.Report).filter(reports_model.Report.report_type == report_type))
        return result.scalars().all()
    except Exception as e:
        raise Exception(f"Error retrieving reports by type {report_type}: {e}")

async def update_report(db: AsyncSession, report_id: int, report: reports_schema.ReportUpdate):
    try:
        result = await db.execute(select(reports_model.Report).filter(reports_model.Report.report_id == report_id))
        db_report = result.scalar_one_or_none()
        
        if db_report:
            if report.report_data:
                db_report.report_data = report.report_data
            if report.report_type:
                db_report.report_type = report.report_type
            await db.commit()
            await db.refresh(db_report)
            return db_report
        else:
            raise Exception(f"Report with ID {report_id} not found.")
    except Exception as e:
        await db.rollback()
        raise Exception(f"Error updating report with ID {report_id}: {e}")

async def get_all_reports(db: AsyncSession, skip: int = 0, limit: int = 100):
    try:
        result = await db.execute(select(reports_model.Report).offset(skip).limit(limit))
        return result.scalars().all()
    except Exception as e:
        raise Exception(f"Error retrieving all reports: {e}")

async def delete_report(db: AsyncSession, report_id: int):
    try:
        result = await db.execute(select(reports_model.Report).filter(reports_model.Report.report_id == report_id))
        db_report = result.scalar_one_or_none()
        
        if db_report:
            await db.delete(db_report)
            await db.commit()
            return db_report
        else:
            raise Exception(f"Report with ID {report_id} not found.")
    except Exception as e:
        await db.rollback()
        raise Exception(f"Error deleting report with ID {report_id}: {e}")
