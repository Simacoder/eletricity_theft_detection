from sqlalchemy.orm import Session
from ..models import alerts as alerts_model
from ..schemas import alerts as alerts_schema
from datetime import datetime

def create_alert(db: Session, alert: alerts_schema.AlertCreate):
    db_alert = alerts_model.Alert(
        anomaly_id=alert.anomaly_id,
        alert_timestamp=datetime.now(),
        alert_message=alert.alert_message,
        alert_severity=alert.alert_severity
    )
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

def get_alerts_by_anomaly_id(db: Session, anomaly_id: int):
    return db.query(alerts_model.Alert).filter(alerts_model.Alert.anomaly_id == anomaly_id).all()

def get_alerts_by_meter_id(db: Session, meter_id: int):
    return db.query(alerts_model.Alert).join(
        alerts_model.Anomaly, alerts_model.Alert.anomaly_id == alerts_model.Anomaly.anomaly_id
    ).filter(alerts_model.Anomaly.meter_id == meter_id).all()

def get_alerts_by_severity(db: Session, severity: str):
    return db.query(alerts_model.Alert).filter(alerts_model.Alert.alert_severity == severity).all()

def update_alert(db: Session, alert_id: int, alert: alerts_schema.AlertUpdate):
    db_alert = db.query(alerts_model.Alert).filter(alerts_model.Alert.alert_id == alert_id).first()
    
    if db_alert:
        db_alert.alert_message = alert.alert_message if alert.alert_message else db_alert.alert_message
        db_alert.alert_severity = alert.alert_severity if alert.alert_severity else db_alert.alert_severity
        db.commit()
        db.refresh(db_alert)
    
    return db_alert

def delete_alert(db: Session, alert_id: int):
    db_alert = db.query(alerts_model.Alert).filter(alerts_model.Alert.alert_id == alert_id).first()
    if db_alert:
        db.delete(db_alert)
        db.commit()
    return db_alert
