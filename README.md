# Inference Engine Learning Project (`sshleifer/tiny-gpt2`)

A minimal, extensible inference engine for learning LLM serving concepts:
- model loading
- generation API
- streaming-ready design
- benchmarking hooks

## Quick Start

1. Create a virtual environment and activate it.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run API:
   - `uvicorn src.infeng.api:app --reload`
4. Open docs:
   - `http://127.0.0.1:8000/docs`

## Project Goals

- Correctness-first text generation using `sshleifer/tiny-gpt2`
- Clear engine abstractions (`engine`, `sampler`, `schemas`)
- Easy path to advanced features: continuous batching, quantization, prefix cache

## Next Milestones

- M1: Single prompt generation + tests
- M2: Batch + streaming
- M3: Profiling and optimization
- M4: Advanced serving features
