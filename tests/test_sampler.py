import pytest
import torch

from infeng.sampler import (
    SamplingParams,
    apply_temperature,
    sample_next_token,
    top_k_filter,
    top_p_filter,
    validate_sampling_params,
)

# probabilities 0.5, 0.25, 0.15, 0.10 -> easy to reason about by hand
PROBS = torch.tensor([[0.5, 0.25, 0.15, 0.10]])
LOGITS = PROBS.log()


def kept(logits: torch.Tensor) -> list[int]:
    return torch.isfinite(logits[0]).nonzero().flatten().tolist()


def test_top_k_keeps_k_largest():
    assert kept(top_k_filter(LOGITS, 2)) == [0, 1]


def test_top_k_zero_keeps_everything():
    assert kept(top_k_filter(LOGITS, 0)) == [0, 1, 2, 3]


def test_top_p_keeps_smallest_set_reaching_p():
    # 0.5 < 0.7, 0.5 + 0.25 = 0.75 >= 0.7 -> keep the first two
    assert kept(top_p_filter(LOGITS, 0.7)) == [0, 1]
    # tiny p still keeps the single best token
    assert kept(top_p_filter(LOGITS, 0.01)) == [0]


def test_top_p_works_on_unsorted_vocab():
    shuffled = LOGITS[:, [2, 0, 3, 1]]  # best token is now at index 1
    assert kept(top_p_filter(shuffled, 0.7)) == [1, 3]


def test_temperature_sharpens_and_flattens():
    cold = torch.softmax(apply_temperature(LOGITS, 0.5), dim=-1)
    hot = torch.softmax(apply_temperature(LOGITS, 2.0), dim=-1)
    assert cold[0, 0] > PROBS[0, 0] > hot[0, 0]


def test_greedy_picks_argmax():
    params = SamplingParams(do_sample=False, top_k=0)
    assert sample_next_token(LOGITS, params).item() == 0


def test_sampling_respects_top_k():
    params = SamplingParams(do_sample=True, top_k=2)
    generator = torch.Generator().manual_seed(0)
    draws = {sample_next_token(LOGITS, params, generator).item() for _ in range(200)}
    assert draws <= {0, 1}


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(max_new_tokens=0),
        SamplingParams(temperature=0),
        SamplingParams(top_p=0),
        SamplingParams(top_k=-1),
        SamplingParams(do_sample=False, top_k=50),  # greedy with top_k set makes no sense
    ],
)
def test_invalid_params(params):
    with pytest.raises(ValueError):
        validate_sampling_params(params)
