from typing import List
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    prompt: str = Field(
        ..., 
        json_schema_extra={"example": "Please email support@enterprise.com with key sk-live-9988776655"},
        description="The text prompt to evaluate for PII and sensitive data exposure."
    )

class DetectedEntity(BaseModel):
    token: str
    entity_type: str

class AnalyzeResponse(BaseModel):
    prompt: str
    contains_pii: bool
    risk_score: float
    latency_ms: float
    detected_entities: List[DetectedEntity]