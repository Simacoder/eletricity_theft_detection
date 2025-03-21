from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from ..models import settings as settings_model
from ..schemas import settings as settings_schema
from sqlalchemy.exc import SQLAlchemyError


async def create_or_update_settings(db: AsyncSession, user_id: int, settings: settings_schema.UserSettingsCreate):
    try:
        result = await db.execute(select(settings_model.Settings).filter(settings_model.Settings.user_id == user_id))
        db_settings = result.scalar_one_or_none()
        
        if db_settings:
            db_settings.notification_email = settings.notification_email
            db_settings.notification_sms = settings.notification_sms
            db_settings.daily_report = settings.daily_report
            await db.commit()
            await db.refresh(db_settings)
        else:
            db_settings = settings_model.Settings(
                user_id=user_id,
                notification_email=settings.notification_email,
                notification_sms=settings.notification_sms,
                daily_report=settings.daily_report
            )
            async with db.begin():
                db.add(db_settings)
            await db.commit()
            await db.refresh(db_settings)
        
        return db_settings
    except SQLAlchemyError as e:
        await db.rollback()
        raise Exception(f"Error creating or updating user settings: {e}")

async def get_settings_by_user(db: AsyncSession, user_id: int):
    try:
        result = await db.execute(select(settings_model.Settings).filter(settings_model.Settings.user_id == user_id))
        return result.scalar_one_or_none()
    except SQLAlchemyError as e:
        raise Exception(f"Error retrieving settings for user ID {user_id}: {e}")

async def get_all_settings(db: AsyncSession, skip: int = 0, limit: int = 100):
    try:
        result = await db.execute(select(settings_model.Settings).offset(skip).limit(limit))
        return result.scalars().all()
    except SQLAlchemyError as e:
        raise Exception(f"Error retrieving all settings: {e}")

async def delete_settings(db: AsyncSession, user_id: int):
    try:
        result = await db.execute(select(settings_model.Settings).filter(settings_model.Settings.user_id == user_id))
        db_settings = result.scalar_one_or_none()
        
        if db_settings:
            await db.delete(db_settings)
            await db.commit()
            return db_settings
        else:
            raise Exception(f"Settings for user ID {user_id} not found.")
    except SQLAlchemyError as e:
        await db.rollback()
        raise Exception(f"Error deleting settings for user ID {user_id}: {e}")

async def get_user_settings(db: AsyncSession, user_id: int):
    try:
        result = await db.execute(select(settings_model.Settings).filter(settings_model.Settings.user_id == user_id))
        return result.scalar_one_or_none()
    except SQLAlchemyError as e:
        raise Exception(f"Error retrieving user settings for user ID {user_id}: {e}")

async def update_user_settings(db: AsyncSession, user_id: int, settings_data: dict):
    try:
        result = await db.execute(select(settings_model.Settings).filter(settings_model.Settings.user_id == user_id))
        user_settings = result.scalar_one_or_none()
        
        if user_settings:
            for key, value in settings_data.items():
                setattr(user_settings, key, value)
            await db.commit()
            await db.refresh(user_settings)
            return user_settings
        else:
            raise Exception(f"User settings for user ID {user_id} not found.")
    except SQLAlchemyError as e:
        await db.rollback()
        raise Exception(f"Error updating user settings for user ID {user_id}: {e}")
