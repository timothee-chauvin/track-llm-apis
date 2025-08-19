"""
Code copied or adapted from https://github.com/i-gao/model-equality-testing
"""

from collections.abc import Callable
from functools import cache

import numpy as np
import torch
from tqdm import tqdm


class CompletionSample:
    def __init__(
        self,
        prompts: np.ndarray | torch.Tensor,
        completions: np.ndarray | torch.Tensor,
        m: int,
    ):
        """
        Represents a sample from a DistributionFromDataset object.
        Args:
            prompts: a (N,) tensor or array of prompt indices
            completions: a (N, L) tensor or array of completions, where L is the maximum completion length
                i.e. this is pre-padded. prompts[i] should correspond to the prompt for completions[i]
            m: total number of prompts; used to enforce that the prompt indices are in [0, m)
        """
        self.m = m

        if isinstance(prompts, np.ndarray):
            prompts = torch.from_numpy(prompts)
        if isinstance(completions, np.ndarray):
            completions = torch.from_numpy(completions)

        self.prompt_sample = prompts.clone().detach()
        self.completion_sample = completions.clone().detach()
        if self.completion_sample.ndim == 1:
            self.completion_sample = self.completion_sample.unsqueeze(-1)

        self.L = self.completion_sample.shape[-1]
        self.N = len(self.prompt_sample)
        self.ns = torch.tensor(
            [(self.prompt_sample == i).sum().item() for i in range(m)]
        )  # effective number of completions for each prompt
        assert self.completion_sample.ndim == 2
        assert self.prompt_sample.ndim == 1

    @property
    def shape(self):
        return (self.m, self.ns.tolist(), self.L)

    @property
    @cache
    def sample(self):
        """(m, n, L) tensor or len-m list of (ni, L) tensors"""
        return self._prompt_completion_to_sample(self.prompt_sample, self.completion_sample)

    @property
    @cache
    def sequences(self):
        return torch.cat([self.prompt_sample.unsqueeze(1), self.completion_sample], dim=1)

    def __str__(self):
        return f"CompletionSample with n={self.ns}"

    def __repr__(self):
        return str(self.sequences)

    @cache
    def _prompt_completion_to_sample(self, prompt_sample, completion_sample):
        """
        Converts view 1 to view 2
        """
        assert len(prompt_sample) == len(completion_sample) == self.N
        max_n = max(self.ns)
        if all([ni == max_n for ni in self.ns]):
            sample = torch.zeros(self.m, max_n, self.shape[-1], dtype=int)
        else:
            sample = [torch.zeros(ni, self.shape[-1], dtype=int) for ni in self.ns]
        count_up = torch.zeros(self.m, dtype=int)
        for i, (prompt, completion) in enumerate(zip(prompt_sample, completion_sample)):
            sample[prompt][count_up[prompt].item()] = completion
            count_up[prompt] += 1
        return sample

    def _sample_to_prompt_completion(self, sample):
        """
        Converts view 2 to view 1
        """
        assert ndim(sample) == 3
        assert len(sample) == self.m
        if isinstance(sample, torch.Tensor):
            indices = []
            for i, tensor in enumerate(sample):
                indices.extend([i] * len(tensor))
            indices = torch.tensor(indices)
            completion_sample = sample.view(-1, sample.shape[-1])
        else:
            indices = []
            for i, tensor in enumerate(sample):
                indices.extend([i] * len(tensor))
            indices = torch.tensor(indices)
            completion_sample = torch.cat(sample)
        return indices, completion_sample


class EmpiricalPvalueCalculator:
    """
    Given an empirical sample of test statitics, provides a callable that returns the p-value of an observed test statistic.
    """

    def __init__(self, observed_stats: np.ndarray):
        """
        Args:
            observed_stats: a numpy array of shape (b,)
                where b is the number of bootstrap samples
        """
        self.stats = observed_stats

    def __call__(self, obs_stat: float | np.ndarray | torch.Tensor) -> float:
        # handle obs_stat: make sure it's a float
        if isinstance(obs_stat, torch.Tensor | np.ndarray):
            obs_stat = obs_stat.item()

        # compare to self.stats and average across the batch dimension (b)
        return np.mean((self.stats >= obs_stat), axis=0).item()


def two_sample_permutation_pvalue(
    sample1: CompletionSample,
    sample2: CompletionSample,
    b=1000,
    return_stats=False,
) -> EmpiricalPvalueCalculator | tuple[EmpiricalPvalueCalculator, np.ndarray]:
    """
    Simulates the empirical distribution of the test statistic by repeatedly permuting the labels
    of the samples and computing the test statistic.
    Args:
        sample1: the first sample
        sample2: the second sample
        b: the number of times to draw samples and compute the test statistic
        return_stats: whether to return the raw test statistics, in addition to
            the p-value calculator
    """
    stats = []
    all_samples = torch.cat(
        [
            sample1.sequences,
            sample2.sequences,
        ],
        dim=0,
    )
    for _ in tqdm(range(b), desc="Permutation bootstrap"):
        ix = torch.randperm(len(all_samples))
        permuted_sample1 = CompletionSample(
            prompts=all_samples[ix][: sample1.N, 0],
            completions=all_samples[ix][: sample1.N, 1:],
            m=sample1.m,
        )
        permuted_sample2 = CompletionSample(
            prompts=all_samples[ix][sample1.N :, 0],
            completions=all_samples[ix][sample1.N :, 1:],
            m=sample1.m,
        )

        stat = mmd_hamming(permuted_sample1, permuted_sample2)
        stats.append(stat)

    stats = np.array(stats)
    if stats.ndim == 1:
        stats = np.expand_dims(stats, 1)
    if stats.ndim == 2:
        stats = np.expand_dims(stats, 2)

    get_pvalue = EmpiricalPvalueCalculator(stats)
    if return_stats:
        return get_pvalue, stats
    return get_pvalue


def run_two_sample_test(
    sample: CompletionSample,
    other_sample: CompletionSample,
    b=1000,
) -> tuple[float, float]:
    """
    Tests whether the samples are drawn from the same distribution
    Args:
        sample: CompletionSample
        other_sample: CompletionSample
        b: int
            Number of bootstrap samples
    Returns:
        pvalue: float
        statistic: float
    """
    get_pvalue = two_sample_permutation_pvalue(sample, other_sample, b=b)
    statistic = mmd_hamming(sample, other_sample)
    pvalue = get_pvalue(statistic)
    return (pvalue, statistic)


### from tests.py
def _mmd(
    X: np.ndarray,
    Y: np.ndarray,
    get_kernel: Callable,
    normalize=True,
) -> float:
    """
    Helper function to compute MMD test statistic.
    Handles normalization.
    Args:
        X: (n, L+1) numpy array of n sequences of length L.
            The first column X[:, 0] is an integer indicating the prompt.
        Y: (m, L+1) numpy array of m sequences of length L.
            The first column Y[:, 0] is an integer indicating the prompt.
        get_kernel: function that computes the kernel matrices K_XX, K_XY, K_YY given X, Y
        normalize: whether to normalize the kernel matrices
    Returns:
        MMD test statistic
    """
    # Create a mask that is True if the prompts are different
    # When computing the kernel, we will zero out the entries where the mask is True
    # since we define the kernel to be 0 when the prompts are different
    prompts_x, prompts_y = X[:, 0], Y[:, 0]
    mask_XY = prompts_x[:, None] != prompts_y[None, :]
    mask_XX = prompts_x[:, None] != prompts_x[None, :]
    mask_YY = prompts_y[:, None] != prompts_y[None, :]

    # Call get_kernel to compute the kernel matrices
    K_XX, K_XY, K_YY = get_kernel(
        X[:, 1:], Y[:, 1:], mask_XX, mask_XY, mask_YY
    )  # remove prompt from seq
    n_XX, n_XY, n_YY = K_XX.size, K_XY.size, K_YY.size

    # Zero out sequences from different prompts according to the mask
    K_XY[mask_XY] = 0
    n_XY -= mask_XY.sum()
    K_XX[mask_XX] = 0
    n_XX -= mask_XX.sum()
    K_YY[mask_YY] = 0
    n_YY -= mask_YY.sum()

    # Normalize the kernel matrices s.t. diagonal is 1
    if normalize:
        # kernel'[x, y] = kernel[x, y] / sqrt(kernel[x, x] * kernel[y, y])
        diagX = np.sqrt(np.diag(K_XX))
        diagY = np.sqrt(np.diag(K_YY))
        diagX[diagX == 0] = 1
        diagY[diagY == 0] = 1
        K_XX /= np.outer(diagX, diagX)
        K_YY /= np.outer(diagY, diagY)
        K_XY /= np.outer(diagX, diagY)

    # Zero out samples with themselves
    np.fill_diagonal(K_XX, 0)
    n_XX -= len(K_XX)
    np.fill_diagonal(K_YY, 0)
    n_YY -= len(K_YY)

    # Compute empirical MMD estimate
    return np.sum(K_XX) / n_XX - 2 * np.sum(K_XY) / n_XY + np.sum(K_YY) / n_YY


def mmd_hamming(
    sample1: CompletionSample,
    sample2: CompletionSample,
) -> float:
    """
    MMD test statistic using K(x, y) = sum_i^L 1[x_i == y_i],
    i.e. whether the marginal densities match
    """

    def get_hamming_kernel(
        X: np.ndarray, Y: np.ndarray, *args, memory_threshold: int = 10000, **kwargs
    ):
        """
        Args:
            X: (n, L) numpy array of n sequences of length L
            Y: (m, L) numpy array of m sequences of length L
        Returns:
            K(X, X), K(X, Y), K(Y, Y) as a tuple
        """
        n, L = X.shape
        m, _ = Y.shape
        max_size_XX = n * n
        max_size_XY = n * m
        max_size_YY = m * m
        if max(max_size_XX, max_size_XY, max_size_YY) <= memory_threshold**2:
            K_XX = np.sum(X[:, None, :] == X[None, :, :], axis=-1).astype(float)
            K_XY = np.sum(X[:, None, :] == Y[None, :, :], axis=-1).astype(float)
            K_YY = np.sum(Y[:, None, :] == Y[None, :, :], axis=-1).astype(float)
        else:
            print("To save memory, computing Hamming using for loops")
            K_XX = np.zeros((n, n), dtype=float)
            K_XY = np.zeros((n, m), dtype=float)
            K_YY = np.zeros((m, m), dtype=float)
            for i in range(n):
                for j in range(i, n):
                    K_XX[i, j] = K_XX[j, i] = np.sum(X[i] == X[j])
            for i in range(n):
                for j in range(m):
                    K_XY[i, j] = K_XY[j, i] = np.sum(X[i] == Y[j])
            for i in range(m):
                for j in range(i, m):
                    K_YY[i, j] = K_YY[j, i] = np.sum(Y[i] == Y[j])
        return K_XX, K_XY, K_YY

    return _mmd(
        X=sample1.sequences.numpy(),
        Y=sample2.sequences.numpy(),
        get_kernel=get_hamming_kernel,
    )


### from utils.py


def ndim(p):
    """
    Args:
        p: either a tensor or a list of tensors or a list of lists of tensors
    """
    if isinstance(p, torch.Tensor):
        return p.ndim
    if not isinstance(p, list | np.ndarray):
        return 0
    elif len(p) > 0:
        return ndim(p[0]) + 1
    else:
        return 1
