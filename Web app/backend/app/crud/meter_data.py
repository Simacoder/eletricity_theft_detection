from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from ..models import meter_data as meter_data_model
from backend.app.schemas.meter_data import MeterDataCreate, MeterDataUpdate
from datetime import datetime

async def create_meter_data(db: AsyncSession, meter_data: MeterDataCreate):
    try:
        db_meter_data = meter_data_model.MeterData(
            meter_id=meter_data.meter_id,
            reading_time=meter_data.reading_time,
            value=meter_data.value,
            consumption_pattern=meter_data.consumption_pattern
        )
        db.add(db_meter_data)
        await db.commit()
        await db.refresh(db_meter_data)
        return db_meter_data
    except Exception as e:
        await db.rollback()
        raise Exception(f"Error creating meter data: {e}")

async def get_meter_data_by_id(db: AsyncSession, meter_id: int):
    try:
        result = await db.execute(select(meter_data_model.MeterData).filter(meter_data_model.MeterData.meter_id == meter_id))
        return result.scalars().all()
    except Exception as e:
        raise Exception(f"Error retrieving meter data by meter_id: {e}")

async def get_all_meter_data(db: AsyncSession, skip: int = 0, limit: int = 100):
    try:
        result = await db.execute(select(meter_data_model.MeterData).offset(skip).limit(limit))
        return result.scalars().all()
    except Exception as e:
        raise Exception(f"Error retrieving all meter data: {e}")

async def get_meter_data_by_time_range(db: AsyncSession, start_time: datetime, end_time: datetime):
    try:
        result = await db.execute(
            select(meter_data_model.MeterData).filter(
                meter_data_model.MeterData.reading_time >= start_time,
                meter_data_model.MeterData.reading_time <= end_time
            )
        )
        return result.scalars().all()
    except Exception as e:
        raise Exception(f"Error retrieving meter data by time range: {e}")

async def get_latest_meter_data(db: AsyncSession, meter_id: int):
    try:
        result = await db.execute(
            select(meter_data_model.MeterData).filter(meter_data_model.MeterData.meter_id == meter_id)
            .order_by(meter_data_model.MeterData.reading_time.desc())
        )
        return result.scalars().first()
    except Exception as e:
        raise Exception(f"Error retrieving latest meter data for meter_id {meter_id}: {e}")

async def update_meter_data(db: AsyncSession, meter_id: int, meter_data: MeterDataUpdate):
    try:
        result = await db.execute(select(meter_data_model.MeterData).filter(meter_data_model.MeterData.meter_id == meter_id))
        db_meter_data = result.scalar_one_or_none()

        if db_meter_data:
            db_meter_data.reading_time = meter_data.reading_time if meter_data.reading_time else db_meter_data.reading_time
            db_meter_data.value = meter_data.value if meter_data.value else db_meter_data.value
            db_meter_data.consumption_pattern = meter_data.consumption_pattern if meter_data.consumption_pattern else db_meter_data.consumption_pattern
            await db.commit()
            await db.refresh(db_meter_data)
            return db_meter_data
        else:
            raise Exception(f"Meter data with meter_id {meter_id} not found.")
    except Exception as e:
        await db.rollback()
        raise Exception(f"Error updating meter data: {e}")

async def delete_meter_data(db: AsyncSession, meter_id: int):
    try:
        result = await db.execute(select(meter_data_model.MeterData).filter(meter_data_model.MeterData.meter_id == meter_id))
        db_meter_data = result.scalar_one_or_none()
        
        if db_meter_data:
            await db.delete(db_meter_data)
            await db.commit()
            return db_meter_data
        else:
            raise Exception(f"Meter data with meter_id {meter_id} not found.")
    except Exception as e:
        await db.rollback()
        raise Exception(f"Error deleting meter data: {e}")
