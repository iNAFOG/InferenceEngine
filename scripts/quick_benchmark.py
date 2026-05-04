from __future__ import annotations

import time

from infeng.config import EngineConfig
from infeng.engine import InferenceEngine
from infeng.sampler import SamplingParams


def main() -> None:
    engine = InferenceEngine(EngineConfig())
    prompts = [
        "Write one sentence about machine learning.",
        "Explain inference engines in simple terms.",
        "What is token sampling?",
    ]

    params = SamplingParams(max_new_tokens=32, do_sample=False)
    latencies = []

    for prompt in prompts:
        start = time.perf_counter()
        result = engine.generate(prompt, params)
        latencies.append((time.perf_counter() - start) * 1000)
        print(f"Prompt: {prompt}")
        print(f"Latency: {result['latency_ms']} ms")
        print(f"Generated tokens: {result['generated_tokens']}\n")

    avg = sum(latencies) / len(latencies)
    print(f"Average wall latency: {avg:.2f} ms")


if __name__ == "__main__":
    main()
