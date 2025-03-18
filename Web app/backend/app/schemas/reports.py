from pydantic import BaseModel

class ReportCreate(BaseModel):
    field1: str
    field2: int

class ReportUpdate(BaseModel):
    field1: str
    field2: int
