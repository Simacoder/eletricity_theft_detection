from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship
from ..core.database import Base

class MeterData(Base):
    __tablename__ = "meter_data"

    id = Column(Integer, primary_key=True, index=True)
    meter_id = Column(Integer, ForeignKey("users.id"))
    reading_timestamp = Column(DateTime, index=True)
    value = Column(Float)
    consumption_pattern = Column(String, index=True)
    
    user = relationship("User", back_populates="meter_data")
