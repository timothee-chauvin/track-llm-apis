import random

import numpy as np
import pytest
import torch

from track_llm_apis.sampling.analyze_gao2025 import (
    CompletionSample,
    mmd_hamming,
    mmd_hamming_torch,
    two_sample_permutation_pvalue,
    two_sample_permutation_pvalue_torch,
)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    random.seed(seed)
    np.random.seed(seed)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_mmd(seed):
    """
    Test that the original and GPU-accelerated versions of MMD match.
    """
    sample1_size = 20
    sample2_size = 25
    vocab_size = 100
    completion_length = 50
    n_permutations = 1000
    randint_dtype = torch.int64
    device = "cuda"

    set_seed(seed)

    prompts1 = torch.randint(0, 5, (sample1_size,), device=device, dtype=randint_dtype)
    prompts2 = torch.randint(0, 5, (sample2_size,), device=device, dtype=randint_dtype)

    completions1 = torch.randint(
        0, vocab_size, (sample1_size, completion_length), device=device, dtype=randint_dtype
    )
    completions2 = torch.randint(
        0, vocab_size, (sample2_size, completion_length), device=device, dtype=randint_dtype
    )

    sample1_gpu = CompletionSample(prompts1, completions1)
    sample2_gpu = CompletionSample(prompts2, completions2)
    sample1_cpu = CompletionSample(prompts1.cpu(), completions1.cpu())
    sample2_cpu = CompletionSample(prompts2.cpu(), completions2.cpu())

    set_seed(seed)
    get_pvalue_gpu, stats_gpu = two_sample_permutation_pvalue_torch(
        sample1_gpu, sample2_gpu, b=n_permutations, return_stats=True
    )
    set_seed(seed)
    get_pvalue_cpu, stats_cpu = two_sample_permutation_pvalue(
        sample1_cpu, sample2_cpu, b=n_permutations, return_stats=True
    )

    stat_gpu = mmd_hamming_torch(
        sample1_gpu.sequences.unsqueeze(0), sample2_gpu.sequences.unsqueeze(0)
    ).item()
    stat_cpu = mmd_hamming(sample1_cpu, sample2_cpu)

    pvalue_gpu = get_pvalue_gpu(stat_gpu)
    pvalue_cpu = get_pvalue_cpu(stat_cpu)

    stats_gpu = stats_gpu.cpu().to(torch.float64)
    stats_cpu = torch.from_numpy(stats_cpu.squeeze())
    assert stats_gpu.shape == stats_cpu.shape
    assert torch.allclose(stats_gpu, stats_cpu, atol=1e-6)
    assert abs(stat_gpu - stat_cpu) < 1e-6
    assert abs(pvalue_gpu - pvalue_cpu) < 1e-6
