from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    label : str = Field(..., exmapl="malignant")
    confidence : float = Field(..., ge=0,le=100)