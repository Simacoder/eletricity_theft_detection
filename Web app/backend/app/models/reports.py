from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import relationship
from ..core.database import Base

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    report_type = Column(String, index=True)
    content = Column(Text)
    timestamp = Column(DateTime)
