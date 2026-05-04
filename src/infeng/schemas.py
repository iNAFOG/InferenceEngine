#this is for api boundary validation/correctness
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(32, ge=1, le=512)
    temperature: float = Field(1.0, gt=0.0, le=5.0)
    top_k: int = Field(50, ge=0, le=200)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    do_sample: bool = True
    seed: int | None = None


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    model: str
    latency_ms: float
