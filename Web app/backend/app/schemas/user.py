from ..schemas import user as user_schema
from pydantic import BaseModel
from sqlalchemy.orm import Session

def create_user(db: Session, user):
    from backend.app.schemas.user import UserCreate
    return UserCreate

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserOut(BaseModel):
    username: str
    email: str

class UserUpdate(BaseModel):
    username: str
    email: str
