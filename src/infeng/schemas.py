#this is for api boundary validation/correctness
from pydantic import BaseModel, Field

from .sampler import SamplingParams


class SamplingFields(BaseModel):
    max_new_tokens: int = Field(32, ge=1, le=512)
    temperature: float = Field(1.0, gt=0.0, le=5.0)
    top_k: int = Field(50, ge=0, le=200)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    do_sample: bool = True
    seed: int | None = None

    def to_sampling_params(self) -> SamplingParams:
        return SamplingParams(**self.model_dump(include=set(SamplingFields.model_fields)))


class GenerateRequest(SamplingFields):
    prompt: str = Field(..., min_length=1)


class BatchGenerateRequest(SamplingFields):
    prompts: list[str] = Field(..., min_length=1, max_length=32)


class GenerateResponse(BaseModel):
    text: str
    completion: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    model: str
    latency_ms: float
    ttft_ms: float
    tokens_per_sec: float


class BatchGenerateResponse(BaseModel):
    results: list[GenerateResponse]
