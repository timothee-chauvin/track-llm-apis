from functools import cache

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from track_llm_apis.config import config
from track_llm_apis.sampling.common import OutputRow, TwoSampleTestResult
from track_llm_apis.util import mmlu_answer_to_choice, mmlu_prompt_to_question


@cache
def is_correct(prompt: str, text: str) -> bool:
    question = mmlu_prompt_to_question(prompt)
    answer = config.sampling.mmlu.answers[question]
    choice = mmlu_answer_to_choice(text)
    return choice == answer


def mmlu_two_sample_test(
    rows_subset: dict[str, list[OutputRow]],
    unchanged_rows_subset: dict[str, list[OutputRow]],
    b: int = 1000,
    compute_pvalue: bool = True,
    **kwargs,
) -> TwoSampleTestResult:
    assert set(rows_subset.keys()) == set(unchanged_rows_subset.keys())
    prompts = list(rows_subset.keys())
    correct1: Bool[Tensor, "N1 P"] = torch.tensor(
        [[is_correct(prompt, row.text[0]) for row in rows_subset[prompt]] for prompt in prompts],
        dtype=torch.bool,
        device=config.analysis.device,
    ).T
    correct2: Bool[Tensor, "N2 P"] = torch.tensor(
        [
            [is_correct(prompt, row.text[0]) for row in unchanged_rows_subset[prompt]]
            for prompt in prompts
        ],
        dtype=torch.bool,
        device=config.analysis.device,
    ).T
    statistic = mmlu_two_sample_statistic(correct1.unsqueeze(0), correct2.unsqueeze(0)).item()
    if compute_pvalue:
        permutation_stats = mmlu_two_sample_permutation_pvalue(correct1, correct2, b=b)
        pvalue = torch.mean(permutation_stats >= statistic, dim=0, dtype=torch.float32).item()
        return TwoSampleTestResult(pvalue=pvalue, statistic=statistic)
    else:
        return TwoSampleTestResult(statistic=statistic)


def mmlu_two_sample_permutation_pvalue(
    correct1: Bool[Tensor, "N1 P"],
    correct2: Bool[Tensor, "N2 P"],
    b: int = 1000,
) -> Float[Tensor, " b"]:
    all_samples = torch.cat([correct1, correct2], dim=0)
    N1 = correct1.shape[0]
    N2 = correct2.shape[0]
    P = correct1.shape[1]
    N_total = N1 + N2
    rand = torch.rand(b, N_total, device=config.analysis.device)
    perm_indices = torch.argsort(rand, dim=1)
    permuted_samples = all_samples[perm_indices]
    assert permuted_samples.shape == (b, N_total, P)
    permuted_t1 = permuted_samples[:, :N1, :]
    permuted_t2 = permuted_samples[:, N1:, :]
    assert permuted_t1.shape == (b, N1, P)
    assert permuted_t2.shape == (b, N2, P)
    stats = mmlu_two_sample_statistic(permuted_t1, permuted_t2)
    return stats


def mmlu_two_sample_statistic(
    correct1: Bool[Tensor, "b N1 P"],
    correct2: Bool[Tensor, "b N2 P"],
) -> Float[Tensor, " b"]:
    # a1, a2: contain the average accuracy for each prompt
    a1: Float[Tensor, "b P"] = correct1.float().mean(dim=1)
    a2: Float[Tensor, "b P"] = correct2.float().mean(dim=1)
    return (a1 - a2).abs().mean(dim=1)
