from sqlalchemy.orm import Session
from ..models import meter_data as meter_data_model
from ..schemas import meter_data as meter_data_schema
from datetime import datetime

def create_meter_data(db: Session, meter_data: meter_data_schema.MeterDataCreate):
    db_meter_data = meter_data_model.MeterData(
        meter_id=meter_data.meter_id,
        reading_time=meter_data.reading_time,
        value=meter_data.value,
        consumption_pattern=meter_data.consumption_pattern
    )
    db.add(db_meter_data)
    db.commit()
    db.refresh(db_meter_data)
    return db_meter_data

def get_meter_data_by_meter_id(db: Session, meter_id: int):
    return db.query(meter_data_model.MeterData).filter(meter_data_model.MeterData.meter_id == meter_id).all()

def get_meter_data_by_time_range(db: Session, start_time: datetime, end_time: datetime):
    return db.query(meter_data_model.MeterData).filter(
        meter_data_model.MeterData.reading_time >= start_time,
        meter_data_model.MeterData.reading_time <= end_time
    ).all()

def get_latest_meter_data(db: Session, meter_id: int):
    return db.query(meter_data_model.MeterData).filter(meter_data_model.MeterData.meter_id == meter_id).order_by(
        meter_data_model.MeterData.reading_time.desc()).first()

def update_meter_data(db: Session, meter_id: int, meter_data: meter_data_schema.MeterDataUpdate):
    db_meter_data = db.query(meter_data_model.MeterData).filter(meter_data_model.MeterData.meter_id == meter_id).first()
    
    if db_meter_data:
        db_meter_data.reading_time = meter_data.reading_time if meter_data.reading_time else db_meter_data.reading_time
        db_meter_data.value = meter_data.value if meter_data.value else db_meter_data.value
        db_meter_data.consumption_pattern = meter_data.consumption_pattern if meter_data.consumption_pattern else db_meter_data.consumption_pattern
        db.commit()
        db.refresh(db_meter_data)
    
    return db_meter_data

def delete_meter_data(db: Session, meter_id: int):
    db_meter_data = db.query(meter_data_model.MeterData).filter(meter_data_model.MeterData.meter_id == meter_id).first()
    if db_meter_data:
        db.delete(db_meter_data)
        db.commit()
    return db_meter_data
