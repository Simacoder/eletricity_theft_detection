from pydantic import BaseModel

class UserSettingsBase(BaseModel):
    setting1: str
    setting2: int

class UserSettingsCreate(UserSettingsBase):
    pass

class UserSettingsUpdate(UserSettingsBase):
    pass
