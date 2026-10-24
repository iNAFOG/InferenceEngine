from dataclasses import replace

import pytest
import torch

from infeng.config import EngineConfig
from infeng.engine import InferenceEngine
from infeng.sampler import SamplingParams

MODEL = "sshleifer/tiny-gpt2"
GREEDY = SamplingParams(max_new_tokens=16, do_sample=False, top_k=0)


@pytest.fixture(scope="module")
def engine():
    # loading the model is the slow part, so all tests in this file share one engine
    return InferenceEngine(EngineConfig(model_name=MODEL, device="cpu"))


@pytest.fixture(scope="module")
def engine_no_cache():
    return InferenceEngine(EngineConfig(model_name=MODEL, device="cpu", use_kv_cache=False))


def test_engine_smoke_generation(engine) -> None:
    output = engine.generate("Hello", SamplingParams(max_new_tokens=8, top_k=0, do_sample=False))

    assert isinstance(output["text"], str)
    assert output["prompt_tokens"] > 0
    assert output["total_tokens"] == output["prompt_tokens"] + output["generated_tokens"]
    assert output["ttft_ms"] <= output["latency_ms"]


def test_deterministic_generation(engine):
    params = SamplingParams(
        max_new_tokens=16,
        temperature=1.0,
        do_sample=True,
        seed=42
    )

    output1 = engine.generate("Hello World", params)
    output2 = engine.generate("Hello World", params)

    assert output1["text"] == output2["text"], "Fixed seed should produce identical output"


def test_greedy_ignores_seed(engine):
    output_a = engine.generate("Hello", replace(GREEDY, seed=67))
    output_b = engine.generate("Hello", replace(GREEDY, seed=420))
    assert output_a["text"] == output_b["text"]


def test_invalid_sampling_params_raise(engine):
    with pytest.raises(ValueError):
        engine.generate("Hello", SamplingParams(max_new_tokens=0))


def test_batch_generation(engine):
    outputs = engine.generate_batch(["Hi", "Hello there"], replace(GREEDY, max_new_tokens=4))

    assert len(outputs) == 2
    assert all("text" in o for o in outputs)
    assert all(o["prompt_tokens"] > 0 for o in outputs)


def test_batch_matches_single(engine):
    # left padding + position ids must make a padded row behave like it was alone
    prompts = ["Hi", "The quick brown fox jumps over"]
    batch = engine.generate_batch(prompts, GREEDY)
    for prompt, out in zip(prompts, batch):
        assert out["text"] == engine.generate(prompt, GREEDY)["text"]


def test_kv_cache_matches_no_cache(engine, engine_no_cache):
    # the cache is only a speed trick, it must never change the output
    prompt = "Machine learning is"
    assert engine.generate(prompt, GREEDY)["text"] == engine_no_cache.generate(prompt, GREEDY)["text"]


def test_matches_huggingface_generate(engine):
    # our hand-written greedy loop should give exactly what HF's generate() gives
    inputs = engine.tokenizer("Machine learning is", return_tensors="pt")
    with torch.inference_mode():
        hf_ids = engine.model.generate(
            **inputs, max_new_tokens=16, do_sample=False, pad_token_id=engine.tokenizer.eos_token_id
        )
    hf_text = engine.tokenizer.decode(hf_ids[0], skip_special_tokens=True)
    assert engine.generate("Machine learning is", GREEDY)["text"] == hf_text


def test_ignore_eos_generates_exact_length(engine):
    out = engine.generate("Hello", replace(GREEDY, max_new_tokens=20, ignore_eos=True))
    assert out["generated_tokens"] == 20


def test_stream_matches_generate(engine):
    events = list(engine.stream_generate("Hello", GREEDY))
    streamed = "".join(e["text"] for e in events if e["type"] == "token")

    assert events[-1]["type"] == "done"
    assert streamed == engine.generate("Hello", GREEDY)["completion"]
