#this is for per req generation behavior + the actual token picking logic
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = True
    seed: int | None = None
    ignore_eos: bool = False  # keep going after EOS, used by benchmarks to get a fixed length


def validate_sampling_params(params: SamplingParams) -> None:
    if params.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if params.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not (0.0 < params.top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    if params.top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not params.do_sample and params.temperature != 1.0:
        raise ValueError("temperature must be 1.0 when do_sample=False (greedy)")
    if not params.do_sample and params.top_k != 0:
        raise ValueError("top_k must be 0 when do_sample=False (greedy)")
    if not params.do_sample and params.top_p != 1.0:
        raise ValueError("top_p must be 1.0 when do_sample=False (greedy)")


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # < 1.0 makes the distribution sharper, > 1.0 makes it flatter
    if temperature == 1.0:
        return logits
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k highest logits in each row, set the rest to -inf. k=0 means no filtering."""
    if k == 0 or k >= logits.shape[-1]:
        return logits
    kth_value = torch.topk(logits, k, dim=-1).values[..., -1, None]  # (batch, 1)
    return logits.masked_fill(logits < kth_value, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose probabilities add up to >= p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    # drop a token if the tokens BEFORE it already reach p. shifting by one keeps
    # the token that crosses p, so at least the top-1 token always survives.
    remove_sorted = (cum_probs - torch.softmax(sorted_logits, dim=-1)) >= p

    remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)  # back to vocab order
    return logits.masked_fill(remove, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits: (batch, vocab) for the last position. returns (batch,) token ids."""
    if not params.do_sample:
        return torch.argmax(logits, dim=-1)

    logits = logits.float()  # fp16 softmax can overflow, so sample in fp32
    logits = apply_temperature(logits, params.temperature)
    logits = top_k_filter(logits, params.top_k)
    logits = top_p_filter(logits, params.top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
