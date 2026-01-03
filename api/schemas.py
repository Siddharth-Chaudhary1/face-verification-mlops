from pydantic import BaseModel

class CompareRequest(BaseModel):
    image1: str
    image2: str
    threshold: float = 0.35
