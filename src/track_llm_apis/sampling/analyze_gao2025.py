"""
Code copied or adapted from https://github.com/i-gao/model-equality-testing
"""

from collections.abc import Callable

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from track_llm_apis.config import config
from track_llm_apis.sampling.common import TwoSampleTestResult


class CompletionSample:
    def __init__(
        self,
        prompts: np.ndarray | torch.Tensor,
        completions: np.ndarray | torch.Tensor,
    ):
        """
        Represents a sample from a DistributionFromDataset object.
        Args:
            prompts: a (N,) tensor or array of prompt indices
            completions: a (N, L) tensor or array of completions, where L is the maximum completion length
                i.e. this is pre-padded. prompts[i] should correspond to the prompt for completions[i]
        """
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
        assert self.completion_sample.ndim == 2
        assert self.prompt_sample.ndim == 1
        self.sequences = torch.cat([self.prompt_sample.unsqueeze(1), self.completion_sample], dim=1)

    def __repr__(self):
        return str(self.sequences)


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


class EmpiricalPvalueCalculatorTorch:
    """
    Given an empirical sample of test statitics, provides a callable that returns the p-value of an observed test statistic.
    """

    def __init__(self, observed_stats: Float[Tensor, " b"]):
        """
        Args:
            observed_stats: a torch tensor of shape (b,)
                where b is the number of bootstrap samples
        """
        self.stats = observed_stats

    def __call__(self, obs_stat: float | Float[Tensor, " b"]) -> float:
        return torch.mean((self.stats >= obs_stat), dim=0, dtype=torch.float32).item()


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
    perm_indices = torch.stack([torch.randperm(len(all_samples)) for _ in range(b)])
    for i in tqdm(range(b), desc="Permutation bootstrap"):
        ix = perm_indices[i]
        permuted_sample1 = CompletionSample(
            prompts=all_samples[ix][: sample1.N, 0],
            completions=all_samples[ix][: sample1.N, 1:],
        )
        permuted_sample2 = CompletionSample(
            prompts=all_samples[ix][sample1.N :, 0],
            completions=all_samples[ix][sample1.N :, 1:],
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


def two_sample_permutation_pvalue_torch(
    sample1: CompletionSample,
    sample2: CompletionSample,
    b=1000,
    return_stats=False,
) -> EmpiricalPvalueCalculatorTorch | tuple[EmpiricalPvalueCalculatorTorch, torch.Tensor]:
    device = config.analysis.device
    assert sample1.sequences.device.type == device, f"move sample1 to {device=} before calling"
    assert sample2.sequences.device.type == device, f"move sample2 to {device=} before calling"
    all_samples = torch.cat(
        [
            sample1.sequences,
            sample2.sequences,
        ],
        dim=0,
    )
    N_total, L = all_samples.shape
    N1 = sample1.N
    N2 = sample2.N
    # generate on CPU then move to device, to use the same PRNG as the non-torch version
    # in order to compare results in tests
    perm_indices = torch.stack([torch.randperm(N_total, device="cpu") for _ in range(b)])
    perm_indices = perm_indices.to(device)
    assert perm_indices.shape == (b, N_total)

    # 2. Gather all permuted samples in a single batch
    permuted_samples = all_samples[perm_indices]
    assert permuted_samples.shape == (b, N_total, L)

    # 3. Split into permuted samples 1 and 2 for all batches
    # Shape: (b, N1, L+1) and (b, N2, L+1)
    permuted_sample1_batch = permuted_samples[:, :N1, :]
    permuted_sample2_batch = permuted_samples[:, N1:, :]
    assert permuted_sample1_batch.shape == (b, N1, L)
    assert permuted_sample2_batch.shape == (b, N2, L)

    stats = mmd_hamming_torch(permuted_sample1_batch, permuted_sample2_batch)

    get_pvalue = EmpiricalPvalueCalculatorTorch(stats)
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


def run_two_sample_test_torch(
    sample: CompletionSample,
    other_sample: CompletionSample,
    b=1000,
) -> TwoSampleTestResult:
    statistic = mmd_hamming_torch(
        sample.sequences.unsqueeze(0), other_sample.sequences.unsqueeze(0)
    ).item()
    if b > 0:
        get_pvalue = two_sample_permutation_pvalue_torch(sample, other_sample, b=b)
        pvalue = get_pvalue(statistic)
        return TwoSampleTestResult(pvalue=pvalue, statistic=statistic)
    else:
        return TwoSampleTestResult(statistic=statistic)


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
    normalize: bool = True,
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
        normalize=normalize,
    )


def mmd_hamming_torch(
    X: Int[Tensor, "b n Lp1"], Y: Int[Tensor, "b m Lp1"], normalize: bool = True
) -> Float[Tensor, " b"]:
    """
    Computes MMD with Hamming kernel sequentially to save memory.
    Args:
        X: (b, n, L+1) tensor of b batches of n sequences.
        Y: (b, m, L+1) tensor of b batches of m sequences.
        normalize: whether to normalize the kernel.
    Returns:
        A (b,) tensor containing the MMD statistic for each batch.
    """
    b, n, L_plus_1 = X.shape
    _, m, _ = Y.shape
    L = L_plus_1 - 1
    device = X.device

    # Extract prompts and completions
    prompts_x, completions_x = X[:, :, 0], X[:, :, 1:]
    prompts_y, completions_y = Y[:, :, 0], Y[:, :, 1:]

    # --- Normalization setup (Memory efficient) ---
    # The diagonal of a hamming kernel k(x,x) is just the sequence length L.
    # We can compute normalization factors without building the full matrices.
    if normalize:
        diag_x_sqrt = (
            torch.full((b, n), L, dtype=torch.float32, device=device).sqrt_().clamp_(min=1e-8)
        )
        diag_y_sqrt = (
            torch.full((b, m), L, dtype=torch.float32, device=device).sqrt_().clamp_(min=1e-8)
        )

    # --- Term 1: K_XX ---
    K_XX = torch.sum(completions_x.unsqueeze(2) == completions_x.unsqueeze(1), dim=-1).float()
    mask_XX = prompts_x.unsqueeze(2) != prompts_x.unsqueeze(1)
    K_XX.masked_fill_(mask_XX, 0)

    if normalize:
        # Shape of outer product: (b, n, n)
        norm_factor = torch.bmm(diag_x_sqrt.unsqueeze(2), diag_x_sqrt.unsqueeze(1))
        K_XX /= norm_factor

    eye_xx = torch.eye(n, device=device).unsqueeze(0)
    K_XX.masked_fill_(eye_xx.bool(), 0)

    n_XX = n * n - mask_XX.sum(dim=(1, 2)) - n
    term1 = K_XX.sum(dim=(1, 2)) / n_XX.clamp(min=1)

    # Free memory
    del K_XX, mask_XX, eye_xx
    if normalize:
        del norm_factor

    # --- Term 3: K_YY (doing YY before XY is equivalent) ---
    K_YY = torch.sum(completions_y.unsqueeze(2) == completions_y.unsqueeze(1), dim=-1).float()
    mask_YY = prompts_y.unsqueeze(2) != prompts_y.unsqueeze(1)
    K_YY.masked_fill_(mask_YY, 0)

    if normalize:
        norm_factor = torch.bmm(diag_y_sqrt.unsqueeze(2), diag_y_sqrt.unsqueeze(1))
        K_YY /= norm_factor

    eye_yy = torch.eye(m, device=device).unsqueeze(0)
    K_YY.masked_fill_(eye_yy.bool(), 0)

    n_YY = m * m - mask_YY.sum(dim=(1, 2)) - m
    term3 = K_YY.sum(dim=(1, 2)) / n_YY.clamp(min=1)

    # Free memory
    del K_YY, mask_YY, eye_yy
    if normalize:
        del norm_factor

    # --- Term 2: K_XY ---
    K_XY = torch.sum(completions_x.unsqueeze(2) == completions_y.unsqueeze(1), dim=-1).float()
    mask_XY = prompts_x.unsqueeze(2) != prompts_y.unsqueeze(1)
    K_XY.masked_fill_(mask_XY, 0)

    if normalize:
        norm_factor = torch.bmm(diag_x_sqrt.unsqueeze(2), diag_y_sqrt.unsqueeze(1))
        K_XY /= norm_factor

    n_XY = n * m - mask_XY.sum(dim=(1, 2))
    term2 = 2 * K_XY.sum(dim=(1, 2)) / n_XY.clamp(min=1)

    # Free memory
    del K_XY, mask_XY
    if normalize:
        del norm_factor

    return term1 - term2 + term3


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
