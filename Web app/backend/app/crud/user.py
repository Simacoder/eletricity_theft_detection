from sqlalchemy.orm import Session
from ..models import user as user_model
from ..schemas import user as user_schema
from ..core.security import get_password_hash, verify_password

def create_user(db: Session, user: user_schema.UserCreate):
    db_user = user_model.User(
        username=user.username,
        password=get_password_hash(user.password),
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_id(db: Session, user_id: int):
    return db.query(user_model.User).filter(user_model.User.id == user_id).first()

def get_user_by_username(db: Session, username: str):
    return db.query(user_model.User).filter(user_model.User.username == username).first()

def update_user(db: Session, user_id: int, user: user_schema.UserUpdate):
    db_user = db.query(user_model.User).filter(user_model.User.id == user_id).first()
    if db_user:
        db_user.username = user.username if user.username else db_user.username
        db_user.password = get_password_hash(user.password) if user.password else db_user.password
        db.commit()
        db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int):
    db_user = db.query(user_model.User).filter(user_model.User.id == user_id).first()
    if db_user:
        db.delete(db_user)
        db.commit()
    return db_user

def verify_user(db: Session, username: str, password: str):
    db_user = get_user_by_username(db, username)
    if db_user and verify_password(password, db_user.password):
        return db_user
    return None
