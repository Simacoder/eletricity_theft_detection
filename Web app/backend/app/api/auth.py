from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from ..models.user import User
from ..crud.user import create_user, get_user_by_username
from ..core.security import create_access_token, verify_password, get_password_hash
from ..core.database import get_db
from datetime import timedelta

router = APIRouter()

class UserCreate(BaseModel):
    username: str
    password: str
    role: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

@router.post("/api/register", response_model=UserCreate)
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    existing_user = await get_user_by_username(db, username=user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = await create_user(db=db, username=user.username, password=hashed_password, role=user.role)
    return new_user

@router.post("/api/login", response_model=Token)
async def login(user: UserLogin, db: AsyncSession = Depends(get_db)):
    await get_user_by_username(db, username=user.username)
