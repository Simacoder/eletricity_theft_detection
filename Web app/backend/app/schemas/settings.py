from pydantic import BaseModel
from typing import Optional

class UserSettingsCreate(BaseModel):
    notification_email: bool
    notification_sms: bool
    daily_report: bool

    class Config:
        from_attributes = True

class UserSettingsUpdate(BaseModel):
    notification_email: Optional[bool] = None
    notification_sms: Optional[bool] = None
    daily_report: Optional[bool] = None

    class Config:
        from_attributes = True

class UserSettingsResponse(UserSettingsCreate):
    user_id: int
    class Config:
        from_attributes = True
