from pydantic import BaseModel
from typing import Optional

class UserCreate(BaseModel):
    username: str
    password: str
    role: str

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None

    class Config:
        from_attributes = True

class UserResponse(BaseModel):
    id: int
    username: str
    role: str

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    username: str
    password: str

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

    class Config:
        from_attributes = True
