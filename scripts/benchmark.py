"""Benchmarks for the engine.

Measures three things:
  1. KV cache on vs off            (same prompt, greedy)
  2. sequential vs batched requests (batch sizes 1/4/8)
  3. float32 vs float16             (cuda only: speed + peak gpu memory)

Usage:
  python scripts/benchmark.py --model Qwen/Qwen2.5-0.5B --device cuda
  python scripts/benchmark.py --model sshleifer/tiny-gpt2 --device cpu --runs 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from infeng.config import EngineConfig
from infeng.engine import InferenceEngine
from infeng.sampler import SamplingParams

PROMPTS = [
    "Write one sentence about machine learning.",
    "Explain inference engines in simple terms.",
    "What is token sampling?",
    "The history of the printing press began",
    "In Python, a list comprehension is",
    "The three laws of motion state that",
    "A good way to learn linear algebra is",
    "Once upon a time in a small village,",
]


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, round(q / 100 * (len(values) - 1)))
    return values[idx]


def sync(engine: InferenceEngine) -> None:
    if engine.device == "cuda":
        torch.cuda.synchronize()


def bench_single(engine: InferenceEngine, params: SamplingParams, runs: int) -> dict:
    engine.generate(PROMPTS[0], params)  # warmup: first call pays for cuda kernels / allocations
    results = [engine.generate(PROMPTS[0], params) for _ in range(runs)]
    latencies = [r["latency_ms"] for r in results]
    return {
        "tokens_per_sec": round(statistics.mean(r["tokens_per_sec"] for r in results), 1),
        "p50_latency_ms": round(percentile(latencies, 50), 1),
        "p95_latency_ms": round(percentile(latencies, 95), 1),
        "ttft_ms": round(statistics.mean(r["ttft_ms"] for r in results), 1),
        "generated_tokens": results[0]["generated_tokens"],
    }


def bench_batching(engine: InferenceEngine, params: SamplingParams, batch_size: int) -> dict:
    prompts = PROMPTS[:batch_size]
    engine.generate_batch(prompts, params)  # warmup

    sync(engine)
    start = time.perf_counter()
    seq_tokens = sum(engine.generate(p, params)["generated_tokens"] for p in prompts)
    sync(engine)
    seq_time = time.perf_counter() - start

    start = time.perf_counter()  # reset the clock, otherwise batch time includes sequential time
    batch_tokens = sum(r["generated_tokens"] for r in engine.generate_batch(prompts, params))
    sync(engine)
    batch_time = time.perf_counter() - start

    return {
        "batch_size": batch_size,
        "sequential_tokens_per_sec": round(seq_tokens / seq_time, 1),
        "batched_tokens_per_sec": round(batch_tokens / batch_time, 1),
        "speedup": round(seq_time / batch_time, 2),
    }


def bench_dtype(config: EngineConfig, dtype: str, params: SamplingParams, runs: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    engine = InferenceEngine(replace(config, dtype=dtype))
    weights_mb = torch.cuda.memory_allocated() / 2**20
    stats = bench_single(engine, params, runs)
    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    del engine
    return {"dtype": dtype, "weights_mb": round(weights_mb), "peak_mb": round(peak_mb), **stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--out", default="benchmarks")
    args = parser.parse_args()

    config = EngineConfig(model_name=args.model, device=args.device, dtype=args.dtype)
    # greedy so every run produces the same tokens, ignore_eos so every run is exactly
    # max_new_tokens long -> fair comparison
    params = SamplingParams(max_new_tokens=args.max_new_tokens, do_sample=False, top_k=0, ignore_eos=True)

    engine = InferenceEngine(config)
    hardware = torch.cuda.get_device_name() if engine.device == "cuda" else "cpu"
    print(f"model={args.model} device={engine.device} ({hardware}) dtype={args.dtype}\n")

    report: dict = {"model": args.model, "device": engine.device, "hardware": hardware,
                    "dtype": args.dtype, "max_new_tokens": args.max_new_tokens}

    print("--- 1. KV cache ---")
    with_cache = bench_single(engine, params, args.runs)
    engine.config = replace(engine.config, use_kv_cache=False)
    no_cache = bench_single(engine, params, args.runs)
    engine.config = replace(engine.config, use_kv_cache=True)
    report["kv_cache"] = {"on": with_cache, "off": no_cache,
                          "speedup": round(with_cache["tokens_per_sec"] / no_cache["tokens_per_sec"], 2)}
    print(f"on : {with_cache}")
    print(f"off: {no_cache}")
    print(f"speedup: {report['kv_cache']['speedup']}x\n")

    print("--- 2. Batching ---")
    report["batching"] = [bench_batching(engine, params, bs) for bs in (1, 4, 8)]
    for row in report["batching"]:
        print(row)
    print()

    del engine
    if torch.cuda.is_available() and config.device in ("auto", "cuda"):
        print("--- 3. float32 vs float16 ---")
        report["dtype"] = [bench_dtype(config, d, params, args.runs) for d in ("float32", "float16")]
        for row in report["dtype"]:
            print(row)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{args.model.replace('/', '_')}_{report['device']}.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out_file}")


if __name__ == "__main__":
    main()
