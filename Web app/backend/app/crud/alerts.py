from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime
from ..models import alerts as alerts_model
from ..schemas import alerts as alerts_schema
from sqlalchemy.exc import SQLAlchemyError

async def create_alert(db: AsyncSession, alert: alerts_schema.AlertCreate):
    try:
        db_alert = alerts_model.Alert(
            anomaly_id=alert.anomaly_id,
            alert_timestamp=datetime.now(),
            alert_message=alert.alert_message,
            alert_severity=alert.alert_severity
        )
        db.add(db_alert)
        await db.commit()
        await db.refresh(db_alert)
        return db_alert
    except SQLAlchemyError as e:
        await db.rollback()
        raise Exception(f"Error creating alert: {e}")

async def get_alerts_by_anomaly_id(db: AsyncSession, anomaly_id: int):
    try:
        result = await db.execute(select(alerts_model.Alert).filter(alerts_model.Alert.anomaly_id == anomaly_id))
        return result.scalars().all()
    except SQLAlchemyError as e:
        raise Exception(f"Error retrieving alerts by anomaly_id: {e}")

async def get_alerts_by_meter_id(db: AsyncSession, meter_id: int):
    try:
        result = await db.execute(
            select(alerts_model.Alert)
            .join(alerts_model.Anomaly, alerts_model.Alert.anomaly_id == alerts_model.Anomaly.anomaly_id)
            .filter(alerts_model.Anomaly.meter_id == meter_id)
        )
        return result.scalars().all()
    except SQLAlchemyError as e:
        raise Exception(f"Error retrieving alerts by meter_id: {e}")

async def get_alerts_by_severity(db: AsyncSession, severity: str):
    try:
        result = await db.execute(select(alerts_model.Alert).filter(alerts_model.Alert.alert_severity == severity))
        return result.scalars().all()
    except SQLAlchemyError as e:
        raise Exception(f"Error retrieving alerts by severity: {e}")

async def update_alert(db: AsyncSession, alert_id: int, alert: alerts_schema.AlertUpdate):
    try:
        result = await db.execute(select(alerts_model.Alert).filter(alerts_model.Alert.alert_id == alert_id))
        db_alert = result.scalar_one_or_none()
        
        if db_alert:
            db_alert.alert_message = alert.alert_message if alert.alert_message else db_alert.alert_message
            db_alert.alert_severity = alert.alert_severity if alert.alert_severity else db_alert.alert_severity
            await db.commit()
            await db.refresh(db_alert)
            return db_alert
        else:
            raise Exception(f"Alert with ID {alert_id} not found.")
    except SQLAlchemyError as e:
        await db.rollback()
        raise Exception(f"Error updating alert: {e}")

async def get_all_alerts(db: AsyncSession, skip: int = 0, limit: int = 100):
    try:
        result = await db.execute(select(alerts_model.Alert).offset(skip).limit(limit))
        return result.scalars().all()
    except SQLAlchemyError as e:
        raise Exception(f"Error retrieving all alerts: {e}")

async def delete_alert(db: AsyncSession, alert_id: int):
    try:
        result = await db.execute(select(alerts_model.Alert).filter(alerts_model.Alert.alert_id == alert_id))
        db_alert = result.scalar_one_or_none()
        if db_alert:
            await db.delete(db_alert)
            await db.commit()
            return db_alert
        else:
            raise Exception(f"Alert with ID {alert_id} not found.")
    except SQLAlchemyError as e:
        await db.rollback()
        raise Exception(f"Error deleting alert: {e}")
