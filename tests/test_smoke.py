from src.infeng.config import EngineConfig
from src.infeng.engine import InferenceEngine
from src.infeng.sampler import SamplingParams


def test_engine_smoke_generation() -> None:
    engine = InferenceEngine(EngineConfig(model_name="sshleifer/tiny-gpt2", device="cpu"))
    output = engine.generate("Hello", SamplingParams(max_new_tokens=8, do_sample=False))

    assert isinstance(output["text"], str)
    assert output["prompt_tokens"] > 0
    assert output["total_tokens"] >= output["prompt_tokens"]
