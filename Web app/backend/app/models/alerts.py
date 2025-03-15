from sqlalchemy import Column, Integer, ForeignKey, String, DateTime
from sqlalchemy.orm import relationship
from ..core.database import Base

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    anomaly_id = Column(Integer, ForeignKey("meter_data.id"))
    alert_timestamp = Column(DateTime)
    alert_message = Column(String)
    severity = Column(String)

    meter_data = relationship("MeterData", back_populates="alerts")
