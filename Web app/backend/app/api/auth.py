from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..models.user import User
from ..crud.user import create_user, get_user_by_username
from ..core.security import create_access_token, verify_password, get_password_hash
from ..core.database import get_db
from datetime import timedelta

router = APIRouter()

# Pydantic models for request validation
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

# Register a new user
@router.post("/register", response_model=UserCreate)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if the username already exists
    existing_user = get_user_by_username(db, username=user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    
    # Create a new user in the database
    new_user = create_user(db=db, username=user.username, password=hashed_password, role=user.role)
    return new_user

# Login user and issue JWT token
@router.post("/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    # Get the user from the database
    db_user = get_user_by_username(db, username=user.username)
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token_expires = timedelta(hours=1)
    access_token = create_access_token(data={"sub": db_user.username}, expires_delta=access_token_expires)
    
    return {"access_token": access_token, "token_type": "bearer"}
