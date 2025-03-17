from pydantic import BaseModel

class MeterDataCreate(BaseModel):
    field_1: str
    field_2: int

class MeterDataUpdate(BaseModel):
    field_1: str
    field_2: int 
    