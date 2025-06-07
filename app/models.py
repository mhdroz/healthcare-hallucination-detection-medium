from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Medical question to analyze")
    multi_stage: bool = Field(False, description="Use multi-stage retrieval")
    fact_check: bool = Field(True, description="Enable external fact-checking")
    consistency_tries: int = Field(3, ge=2, le=10, description="Number of consistency checks")


class SafetyResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    safety_score: int
    max_safety_score: int
    attribution_score: float
    consistency_score: float
    semantic_entropy: float
    weak_sentences: list
    has_weak_sentences: bool
    multi_stage_used: bool
    fact_check_enabled: bool
    external_validation: Optional[dict] = None
    safety_interpretations: dict
    source_chunks: list


class EvaluationResponse(BaseModel):
    faithfulness_score: float
    relevancy_score: float
    faithfulness_interpretation: str
    relevancy_interpretation: str
    overall_grade: str
    num_questions: int
    detailed_scores: list


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool