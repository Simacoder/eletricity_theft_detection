from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from ..core.database import Base

class Settings(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    notifications_enabled = Column(Boolean, default=True)
    email_notifications = Column(Boolean, default=True)
    sms_notifications = Column(Boolean, default=False)

    user = relationship("User", back_populates="settings")

