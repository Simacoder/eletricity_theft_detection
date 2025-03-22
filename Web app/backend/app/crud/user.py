from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from ..models import user as user_model
from ..schemas import user as user_schema
from ..core.security import get_password_hash, verify_password
from sqlalchemy.exc import SQLAlchemyError

async def create_user(db: AsyncSession, user: user_schema.UserCreate):
    try:
        db_user = user_model.User(
            
            username=user.username,
            email=user.email,
            password=get_password_hash(user.password),
        )
        async with db.begin():
            db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        return db_user
    except SQLAlchemyError as e:
        await db.rollback()
        raise Exception(f"Error creating user: {e}")

async def get_user_by_id(db: AsyncSession, user_id: int):
    try:
        result = await db.execute(select(user_model.User).filter(user_model.User.id == user_id))
        return result.scalar_one_or_none()
    except SQLAlchemyError as e:
        raise Exception(f"Error retrieving user by ID {user_id}: {e}")

async def get_user_by_username(db: AsyncSession, username: str):
    try:
        result = await db.execute(select(user_model.User).filter(user_model.User.username == username))
        return result.scalar_one_or_none()
    except SQLAlchemyError as e:
        raise Exception(f"Error retrieving user by username {username}: {e}")

async def update_user(db: AsyncSession, user_id: int, user: user_schema.UserUpdate):
    try:
        result = await db.execute(select(user_model.User).filter(user_model.User.id == user_id))
        db_user = result.scalar_one_or_none()
        
        if db_user:
            if user.username:
                db_user.username = user.username
            if user.password:
                db_user.password = get_password_hash(user.password)
            await db.commit()
            await db.refresh(db_user)
            return db_user
        else:
            raise Exception(f"User with ID {user_id} not found.")
    except SQLAlchemyError as e:
        await db.rollback()
        raise Exception(f"Error updating user with ID {user_id}: {e}")

async def delete_user(db: AsyncSession, user_id: int):
    try:
        result = await db.execute(select(user_model.User).filter(user_model.User.id == user_id))
        db_user = result.scalar_one_or_none()
        
        if db_user:
            await db.delete(db_user)
            await db.commit()
            return db_user
        else:
            raise Exception(f"User with ID {user_id} not found.")
    except SQLAlchemyError as e:
        await db.rollback()
        raise Exception(f"Error deleting user with ID {user_id}: {e}")

async def verify_user(db: AsyncSession, username: str, password: str):
    try:
        db_user = await get_user_by_username(db, username)
        if db_user and verify_password(password, db_user.password):
            return db_user
        return None
    except SQLAlchemyError as e:
        raise Exception(f"Error verifying user with username {username}: {e}")
