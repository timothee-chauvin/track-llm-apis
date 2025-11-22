from functools import cache

import torch
from frozendict import frozendict
from frozenlist import FrozenList
from jaxtyping import Float
from torch import Tensor

from track_llm_apis.analyze import ResponseData
from track_llm_apis.config import config
from track_llm_apis.sampling.common import (
    CompressedOutputRow,
    TwoSampleTestResult,
    TwoSampleTestResultWithDate,
)


def frozenlist(li: list) -> FrozenList:
    fl = FrozenList(li)
    fl.freeze()
    return fl


@cache
def build_logprob_tensor_row(
    logprob: frozendict[int | str, float],
    seen_tokens: frozenset[int | str],
    token_short_ids: frozendict[int | str, int],
) -> Float[Tensor, " nt"]:
    t = torch.full(
        (len(seen_tokens),), float("nan"), dtype=torch.float32, device=config.analysis.device
    )
    for token_id in seen_tokens:
        if token_id in logprob:
            t[token_short_ids[token_id]] = logprob[token_id]
        else:
            # Left censoring for the missing tokens
            t[token_short_ids[token_id]] = min(logprob.values())
    assert not torch.any(torch.isnan(t))
    return t


@cache
def build_logprob_tensor_cached_rows(
    logprobs: FrozenList[frozendict[int | str, float]],
    seen_tokens: frozenset[int | str],
    token_short_ids: frozendict[int | str, int],
) -> Float[Tensor, "N nt"]:
    """Best for repeatedly building small tensors for a single hypothesis test"""
    t = torch.stack([build_logprob_tensor_row(lp, seen_tokens, token_short_ids) for lp in logprobs])
    return t


@cache
def build_logprob_tensor_oneshot(
    logprobs: FrozenList[frozendict[int | str, float]],
    seen_tokens: frozenset[int | str],
    token_short_ids: frozendict[int | str, int],
) -> Float[Tensor, "N nt"]:
    """Best when building a big tensor with all the values from e.g. a time series, for multiple hypothesis tests."""
    t = torch.empty(
        (len(logprobs), len(seen_tokens)), dtype=torch.float32, device=config.analysis.device
    )
    row_mins = [min(lp.values()) for lp in logprobs]
    for i, min_val in enumerate(row_mins):
        t[i, :] = min_val
    for i, lp in enumerate(logprobs):
        for token_id, logprob in lp.items():
            t[i, token_short_ids[token_id]] = logprob
    return t


@cache
def get_seen_tokens(
    logprobs: FrozenList[frozendict[int | str, float]],
) -> tuple[frozenset[int | str], frozendict[int | str, int]]:
    """
    Input: a list of logprob dictionaries (token_id -> logprob value)

    Output:
    - the set of all seen token IDs
    - a dictionary mapping each seen token ID to a short ID (0, 1, 2, ...)
    """
    seen_tokens = frozenset(token_id for lp in logprobs for token_id in lp.keys())
    token_short_ids = frozendict({token_id: i for i, token_id in enumerate(seen_tokens)})
    return seen_tokens, token_short_ids


def logprob_time_series(
    data: list[ResponseData], n_per_test: int, pvalue_b: int = 1000
) -> list[TwoSampleTestResultWithDate]:
    """
    Apply logprob tracking on all sets of 2 consecutive windows of size n_per_test."""
    # Convert to common format
    logprobs = frozenlist(
        [frozendict({tok: lp for tok, lp in zip(rd.top_tokens, rd.logprobs)}) for rd in data]
    )
    seen_tokens, token_short_ids = get_seen_tokens(logprobs)
    nt = len(seen_tokens)
    print(f"Number of unique tokens: {nt}")
    all_time_t = build_logprob_tensor_oneshot(logprobs, seen_tokens, token_short_ids)

    n_windows = len(data) - 2 * n_per_test
    indices = torch.arange(n_per_test, len(data) - n_per_test)

    # shape (n_windows, n_per_test, nt)
    t1_stack = torch.stack([all_time_t[i - n_per_test : i] for i in indices])
    t2_stack = torch.stack([all_time_t[i : i + n_per_test] for i in indices])

    # (n_windows,)
    statistics = logprob_two_sample_statistic(t1_stack, t2_stack)

    if pvalue_b == 0:
        return [
            TwoSampleTestResultWithDate(date=data[i].date, statistic=s.item())
            for i, s in zip(indices, statistics)
        ]

    # Split in chunks along the pvalue_b dimension
    # 1. How much memory do we have available?
    free, _ = torch.cuda.mem_get_info()
    # 2. How much memory does one permutation require?
    fp32 = 4
    memory_per_perm = n_windows * (2 * n_per_test) * nt * fp32
    safety_factor = 0.7
    pvalue_chunk = int(free * safety_factor / memory_per_perm)
    assert pvalue_chunk > 0, "Not enough memory to compute even one permutation!"

    if pvalue_chunk < pvalue_b:
        print(
            f"Computing p-values in chunks of {pvalue_chunk} out of {pvalue_b} due to memory constraints."
        )

    all_permuted_stats = []
    for start_idx in range(0, pvalue_b, pvalue_chunk):
        end_idx = min(start_idx + pvalue_chunk, pvalue_b)
        chunk_size = end_idx - start_idx
        permuted_stats = permuted_stats_batched(
            t1_stack,
            t2_stack,
            n_windows,
            n_per_test,
            chunk_size,
        )
        all_permuted_stats.append(permuted_stats)
    all_permuted_stats = torch.cat(all_permuted_stats, dim=1)  # (n_windows, pvalue_b)
    pvalues = (all_permuted_stats >= statistics[:, None]).float().mean(dim=1)  # (n_windows,)

    return [
        TwoSampleTestResultWithDate(date=data[i].date, pvalue=p.item(), statistic=s.item())
        for i, p, s in zip(indices, pvalues, statistics)
    ]


def permuted_stats_batched(
    t1_chunk: Float[Tensor, "n_windows n_per_test nt"],
    t2_chunk: Float[Tensor, "n_windows n_per_test nt"],
    n_windows: int,
    n_per_test: int,
    chunk_size: int,
) -> Float[Tensor, "n_windows chunk_size"]:
    """
    Compute permuted statistics for a chunk of two-sample tests.
    """
    all_samples = torch.cat([t1_chunk, t2_chunk], dim=1)  # (n_windows, 2*n_per_test, nt)
    rand = torch.rand(chunk_size, 2 * n_per_test, device=config.analysis.device)
    # (chunk_size, 2*n_per_test)
    perm_indices = torch.argsort(rand, dim=1)

    # (n_windows, chunk_size, 2*n_per_test, nt)
    repeated_samples = all_samples[:, None, :, :].expand(-1, chunk_size, -1, -1)
    # (n_windows, chunk_size, 2*n_per_test, nt)
    perm_indices_expanded = perm_indices[None, :, :, None].expand(
        n_windows, -1, -1, all_samples.shape[-1]
    )
    permuted_samples = torch.gather(repeated_samples, dim=2, index=perm_indices_expanded)

    # (n_windows, chunk_size, n_per_test, nt)
    perm_t1 = permuted_samples[:, :, :n_per_test, :]
    perm_t2 = permuted_samples[:, :, n_per_test:, :]

    # perm_t1.flatten(0, 1): (n_windows * chunk_size, n_per_test, nt)
    perm_stats = logprob_two_sample_statistic(perm_t1.flatten(0, 1), perm_t2.flatten(0, 1))
    # (n_windows, chunk_size)
    perm_stats = perm_stats.reshape(n_windows, chunk_size)

    return perm_stats


def logprob_two_sample_test_from_compressed_output_row(
    sample1: dict[str, list[CompressedOutputRow]],
    sample2: dict[str, list[CompressedOutputRow]],
    pvalue_b: int = 1000,
    **kwargs,
) -> TwoSampleTestResult:
    """
    This weird function signature is to match the other two-sample test functions.
    """
    assert len(sample1) == 1 and len(sample2) == 1
    values1 = next(iter(sample1.values()))
    values2 = next(iter(sample2.values()))
    lp1 = [r.first_token_logprobs for r in values1]
    lp2 = [r.first_token_logprobs for r in values2]
    all_logprobs = frozenlist([frozendict(lp) for lp in lp1 + lp2])
    seen_tokens, token_short_ids = get_seen_tokens(all_logprobs)
    sample1_logprobs = frozenlist([frozendict(lp) for lp in lp1])
    sample2_logprobs = frozenlist([frozendict(lp) for lp in lp2])
    t1 = build_logprob_tensor_cached_rows(sample1_logprobs, seen_tokens, token_short_ids)
    t2 = build_logprob_tensor_cached_rows(sample2_logprobs, seen_tokens, token_short_ids)
    return logprob_two_sample_test_from_tensors(t1, t2, pvalue_b=pvalue_b)


def logprob_two_sample_test_from_tensors(
    t1: Float[Tensor, "N1 nt"],
    t2: Float[Tensor, "N2 nt"],
    pvalue_b: int = 1000,
) -> TwoSampleTestResult:
    statistic = logprob_two_sample_statistic(t1.unsqueeze(0), t2.unsqueeze(0)).item()
    if pvalue_b > 0:
        permutation_stats = logprob_two_sample_permutation_pvalue(t1, t2, b=pvalue_b)
        pvalue = torch.mean(permutation_stats >= statistic, dim=0, dtype=torch.float32).item()
        return TwoSampleTestResult(pvalue=pvalue, statistic=statistic)
    else:
        return TwoSampleTestResult(statistic=statistic)


def logprob_two_sample_permutation_pvalue(
    t1: Float[Tensor, "N1 nt"],
    t2: Float[Tensor, "N2 nt"],
    b: int = 1000,
) -> Float[Tensor, " b"]:
    all_samples = torch.cat([t1, t2], dim=0)
    N1 = t1.shape[0]
    N2 = t2.shape[0]
    nt = t1.shape[1]
    N_total = N1 + N2
    rand = torch.rand(b, N_total, device=config.analysis.device)
    perm_indices = torch.argsort(rand, dim=1)
    permuted_samples = all_samples[perm_indices]
    assert permuted_samples.shape == (b, N_total, nt)
    permuted_t1 = permuted_samples[:, :N1, :]
    permuted_t2 = permuted_samples[:, N1:, :]
    assert permuted_t1.shape == (b, N1, nt)
    assert permuted_t2.shape == (b, N2, nt)
    stats = logprob_two_sample_statistic(permuted_t1, permuted_t2)
    return stats


def logprob_two_sample_statistic(
    t1: Float[Tensor, "b N1 nt"],
    t2: Float[Tensor, "b N2 nt"],
) -> Float[Tensor, " b"]:
    """
    The statistic is computed as follows:
    Each token t_i has N1 logprobs in t1 and N2 logprobs in t2, call their averages a1_i and a2_i.
    In vector form, we have a1 = (a1_1, ..., a1_nt) and a2 = (a2_1, ..., a2_nt).
    The statistic is the L1 norm of a1 - a2 divided by the number of tokens, i.e.
    the average of the absolute differences between the two averages.
    """
    a1: Float[Tensor, "b nt"] = t1.mean(dim=1)
    a2: Float[Tensor, "b nt"] = t2.mean(dim=1)
    return (a1 - a2).abs().mean(dim=1)
