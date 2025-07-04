from pydantic import BaseModel

class Job(BaseModel):
    id: int
    name: str
    enable: bool
    detail: str
    experience: str