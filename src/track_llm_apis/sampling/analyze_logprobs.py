import torch
from jaxtyping import Float
from torch import Tensor

from track_llm_apis.config import config
from track_llm_apis.sampling.common import OutputRow


def logprob_two_sample_test(
    sample1: list[OutputRow],
    sample2: list[OutputRow],
    b: int = 1000,
) -> tuple[float, float]:
    # Convert the samples to suitable tensors.
    seen_tokens = set(
        token_id for r in sample1 + sample2 for lp in r.logprobs for token_id in lp.keys()
    )
    token_short_ids = {token_id: i for i, token_id in enumerate(seen_tokens)}
    t1 = torch.full(
        (len(sample1), len(seen_tokens)),
        float("nan"),
        dtype=torch.float32,
        device=config.analysis.device,
    )
    t2 = torch.full(
        (len(sample2), len(seen_tokens)),
        float("nan"),
        dtype=torch.float32,
        device=config.analysis.device,
    )
    for source, target in [(sample1, t1), (sample2, t2)]:
        for i, row in enumerate(source):
            assert len(row.logprobs) == 1
            min_logprob = min(row.logprobs[0].values())
            for token_id in seen_tokens:
                if token_id in row.logprobs[0]:
                    target[i, token_short_ids[token_id]] = row.logprobs[0][token_id]
                else:
                    # Left censoring for the missing tokens
                    target[i, token_short_ids[token_id]] = min_logprob
        assert not torch.any(torch.isnan(target))
    permutation_stats = logprob_two_sample_permutation_pvalue(t1, t2, b=b)
    statistic = logprob_two_sample_statistic(t1.unsqueeze(0), t2.unsqueeze(0)).item()
    pvalue = torch.mean(permutation_stats >= statistic, dim=0, dtype=torch.float32).item()
    return pvalue, statistic


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
    perm_indices = torch.stack(
        [torch.randperm(N_total, device=config.analysis.device) for _ in range(b)]
    )
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
