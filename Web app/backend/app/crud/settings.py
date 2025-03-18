from sqlalchemy.orm import Session
from ..models import settings as settings_model
from ..schemas import settings as settings_schema

def create_or_update_settings(db: Session, user_id: int, settings: settings_schema.UserSettingsCreate):
    db_settings = db.query(settings_model.Settings).filter(settings_model.Settings.user_id == user_id).first()
    
    if db_settings:
        db_settings.notification_email = settings.notification_email
        db_settings.notification_sms = settings.notification_sms
        db_settings.daily_report = settings.daily_report
        db.commit()
        db.refresh(db_settings)
    else:
        db_settings = settings_model.Settings(
            user_id=user_id,
            notification_email=settings.notification_email,
            notification_sms=settings.notification_sms,
            daily_report=settings.daily_report
        )
        db.add(db_settings)
        db.commit()
        db.refresh(db_settings)
    
    return db_settings

def get_settings_by_user(db: Session, user_id: int):
    return db.query(settings_model.Settings).filter(settings_model.Settings.user_id == user_id).first()

def get_all_settings(db: Session, skip: int = 0, limit: int = 100):
    return db.query(settings_model.Settings).offset(skip).limit(limit).all()

def delete_settings(db: Session, user_id: int):
    db_settings = db.query(settings_model.Settings).filter(settings_model.Settings.user_id == user_id).first()
    if db_settings:
        db.delete(db_settings)
        db.commit()
    return db_settings

def get_user_settings(db: Session, user_id: int):
    settings = db.query(settings_model.Settings).filter(settings_model.Settings.user_id == user_id).first()
    return settings

def update_user_settings(db: Session, user_id: int, settings_data: dict):
    user_settings = db.query(settings_model.Settings).filter(settings_model.Settings.user_id == user_id).first()
    if user_settings:
        for key, value in settings_data.items():
            setattr(user_settings, key, value)
        db.commit()
        return user_settings
    else:
        return None
