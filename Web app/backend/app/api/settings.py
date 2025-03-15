from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from app.models.settings import UserSettings
from app.crud.settings import get_user_settings, update_user_settings
from ..core.database import get_db

# Initialize the router
router = APIRouter()

# Pydantic models for request validation and responses
class SettingsCreate(BaseModel):
    notification_preference: str
    alert_frequency: str

class SettingsResponse(SettingsCreate):
    user_id: int

# Get user settings
@router.get("/{user_id}", response_model=SettingsResponse)
async def get_user_settings_by_id(user_id: int, db: Session = Depends(get_db)):
    settings = get_user_settings(db, user_id)
    if not settings:
        raise HTTPException(status_code=404, detail="User settings not found")
    return settings

# Update user settings
@router.put("/{user_id}", response_model=SettingsResponse)
async def update_user_settings_by_id(user_id: int, settings_data: SettingsCreate, db: Session = Depends(get_db)):
    updated_settings = update_user_settings(db, user_id, settings_data)
    if not updated_settings:
        raise HTTPException(status_code=400, detail="Error updating user settings")
    return updated_settings
