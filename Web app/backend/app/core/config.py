import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:phandas2024@localhost:5432/db_name"
    DATABASE_NAME: str = "electricity_fraud_detection_db"
    DATABASE_USER: str = "postgres"
    DATABASE_PASSWORD: str = "password"
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432

    SECRET_KEY: str = "data-phandas"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    FRAUD_DETECTION_MODEL_PATH: str = os.path.join(os.getcwd(), "fraud_detection_model.pkl")

    class Config:
        env_file = ".env"

settings = Settings()
