from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api import auth, meter_data, alerts, reports, settings
from .core.database import engine
from .models import Base

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(meter_data.router, prefix="/meter_data", tags=["Meter Data"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(reports.router, prefix="/reports", tags=["Reports"])
app.include_router(settings.router, prefix="/settings", tags=["Settings"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Electricity Fraud Detection!"}
