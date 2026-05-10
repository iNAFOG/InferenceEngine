# Inference Engine Learning Project (`sshleifer/tiny-gpt2`)

A minimal, extensible **inference engine** for learning LLM serving concepts. Currently achieving **727 tokens/sec avg speed** for TinyGPT2 with plans to add more complex serving logic.

**Current Features:**
- model loading
- generation API
- streaming-ready design
- benchmarking hooks

## Quick Start

1. Create a virtual environment and activate it.
2. Install dependencies:
   - `pip install --upgrade pip`
   - `pip install -r requirements.txt`
   - `pip install -e .`
3. Run API:
   - `uvicorn src.infeng.api:app --reload`
4. Open docs:
   - `http://127.0.0.1:8000/docs`
5. For testing using Pytest:
   - `pytest -s tests/test_smoke.py`

## Project Goals

- Correctness-first text generation using `sshleifer/tiny-gpt2`
- Clear engine abstractions (`engine`, `sampler`, `schemas`)
- Foundation for advanced serving patterns

## Planned Advanced Logic

Future releases will implement sophisticated inference optimization techniques:

- **Batch Processing & Continuous Batching**: Group requests for improved throughput and GPU utilization
- **Streaming & Token-Level Serving**: Real-time token streaming to clients for reduced latency
- **Quantization**: Model compression (int8, fp16) for faster inference and lower memory footprint
- **Prefix Cache / KV-Cache Optimization**: Reuse cached key-value tensors for repeated prompts
- **Profiling & Performance Optimization**: Detailed latency analysis and bottleneck identification
