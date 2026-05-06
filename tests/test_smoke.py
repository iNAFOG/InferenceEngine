from infeng.config import EngineConfig
from infeng.engine import InferenceEngine
from infeng.sampler import SamplingParams


def test_engine_smoke_generation() -> None:
    engine = InferenceEngine(EngineConfig(model_name="sshleifer/tiny-gpt2", device="cpu"))
    output = engine.generate("Hello", SamplingParams(max_new_tokens=8, do_sample=False))

    assert isinstance(output["text"], str)
    assert output["prompt_tokens"] > 0
    assert output["total_tokens"] >= output["prompt_tokens"]


def test_deterministic_generation():
    engine = InferenceEngine(EngineConfig(model_name="sshleifer/tiny-gpt2", device="cpu"))
    params = SamplingParams(
        max_new_tokens=16,
        temperature=1.0,
        do_sample=True,
        seed=42
    )

    output1 = engine.generate("Hello World", params)
    output2 = engine.generate("Hello World", params)

    assert output1["text"] == output2["text"], "Fixed seed should produce identical output"