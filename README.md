# InfEng: a small LLM inference engine

A minimal, readable **LLM inference engine** written to learn how LLM serving works.
It runs its **own token-by-token decode loop** (no `model.generate()`), including KV caching,
batched decoding with left padding, top-k / top-p sampling, and token streaming, all behind a FastAPI server.

Works with any Hugging Face causal LM. Tested with `sshleifer/tiny-gpt2` (unit tests) and `Qwen/Qwen2.5-0.5B` (benchmarks).

## Features

- **Hand-written decode loop** with a **KV cache** (prefill once, then feed one token per step).
  Its greedy output matches Hugging Face `generate()` exactly, and runs at the same speed.
- **Own sampler:** temperature, top-k, top-p (nucleus), greedy, and per-request seeded RNG.
- **Batched generation** for prompts of different lengths (left padding + position ids + attention mask).
  Each batched result matches running that prompt alone.
- **Token streaming** over HTTP as NDJSON, with UTF-8 safe incremental decoding.
- **Metrics per request:** latency, time-to-first-token (TTFT), tokens/sec.
- **fp32 / fp16 / bf16** model loading.
- **Validation in two places:** Pydantic at the API boundary, and engine checks that greedy and sampling settings agree.
- **28 pytest tests:** sampler math, determinism, KV-cache vs no-cache equality, batch vs single equality,
  HF parity, streaming, and the API.

## Benchmarks

`Qwen/Qwen2.5-0.5B`, NVIDIA GTX 1650 (4 GB), greedy, 256 new tokens. From `scripts/benchmark.py`.

| Experiment | Result |
|---|---|
| KV cache on vs off | **33.4 vs 13.5 tok/s, 2.47x faster** |
| Batched vs sequential, batch size 4 | 107.1 vs 31.4 tok/s, 3.41x |
| Batched vs sequential, batch size 8 | **176.4 vs 31.1 tok/s, 5.67x** |
| fp16 vs fp32 weights | **958 vs 1894 MB (about half)**, but 22.4 vs 30.8 tok/s |
| Time to first token | about 31 ms |

fp16 is slower here because the GTX 1650 has no tensor cores. It still halves memory.

## Quick Start

1. Create a virtual environment and activate it.
2. Install dependencies:
   - `pip install --upgrade pip`
   - `pip install -r requirements.txt`
   - `pip install -e .`
3. Run API (pick the model / device / dtype with env vars, all optional):
   - `$env:INFENG_MODEL="Qwen/Qwen2.5-0.5B"; $env:INFENG_DEVICE="cuda"`
   - `uvicorn infeng.api:app`
4. Open docs:
   - `http://127.0.0.1:8000/docs`
5. Try streaming:
   - `python scripts/stream_client.py`
6. Tests:
   - `pytest`
7. Benchmarks:
   - `python scripts/benchmark.py --model Qwen/Qwen2.5-0.5B --device cuda --max-new-tokens 256`

## API

| Endpoint | What it does |
|---|---|
| `GET /health` | model, device, dtype, kv cache on/off |
| `POST /generate` | one prompt → full result + metrics |
| `POST /generate/batch` | list of prompts → one batched forward pass per step |
| `POST /generate/stream` | one prompt → NDJSON lines: `{"type":"token",...}` ... `{"type":"done",...}` |

## Layout

```
src/infeng/
  config.py    engine settings (model, device, dtype, kv cache)
  sampler.py   SamplingParams, validation, temperature / top-k / top-p / sampling
  engine.py    model loading + the decode loop (single, batch, stream)
  schemas.py   API request / response models
  api.py       FastAPI app
scripts/       benchmark.py, stream_client.py
tests/         test_sampler.py, test_smoke.py, test_api.py
```

## Next

- Continuous batching: let new requests join a running batch between decode steps
- Prefix caching: reuse KV cache for prompts that share a prefix
- int8 quantization
- Profiling where the time per decode step goes
