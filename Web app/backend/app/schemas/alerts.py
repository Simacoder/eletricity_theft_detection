from pydantic import BaseModel

class AlertCreate(BaseModel):
    field1: str
    field2: int
    
class AlertUpdate(BaseModel):
    field1: str
    field2: int