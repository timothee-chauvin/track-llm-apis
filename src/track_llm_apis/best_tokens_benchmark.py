"""Benchmark for finding input tokens with specific logprob characteristics."""

import logging
import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import fire
import numpy as np
import orjson
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from track_llm_apis.config import config
from track_llm_apis.util import slugify

logger = logging.getLogger("track-llm-apis")

BENCHMARK_DIR = config.data_dir / "best_tokens_benchmark"


def _benchmark_path(model_name: str) -> Path:
    return BENCHMARK_DIR / f"{slugify(model_name)}.json"


def create_benchmark(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    cumulative_prob_threshold: float = 0.995,
    batch_size: int = 512,
    max_logprobs: int = 100,
) -> Path:
    """Extract top logprobs for all single-token prompts."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output_file = _benchmark_path(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocabulary size: {vocab_size}")

    results = {}
    total_batches = (vocab_size + batch_size - 1) // batch_size

    logger.info(f"Processing {vocab_size} tokens in {total_batches} batches...")

    for batch_idx in tqdm(range(total_batches), desc="Batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, vocab_size)
        batch_token_ids = list(range(start_idx, end_idx))
        actual_batch_size = len(batch_token_ids)

        input_ids = torch.tensor([[tid] for tid in batch_token_ids], device=device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            logprobs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            k = min(max_logprobs, probs.shape[-1])
            top_probs, top_indices = torch.topk(probs, k=k, dim=-1, sorted=False)
            top_logprobs = torch.gather(logprobs, dim=-1, index=top_indices)

            sorted_probs, sort_order = torch.sort(top_probs, dim=-1, descending=True)
            sorted_indices = torch.gather(top_indices, dim=-1, index=sort_order)
            sorted_log_probs = torch.gather(top_logprobs, dim=-1, index=sort_order)

            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            n_tokens_per_item = (cumsum_probs <= cumulative_prob_threshold).sum(dim=-1) + 1
            n_tokens_per_item = torch.clamp(n_tokens_per_item, max=k)
            max_n_tokens = int(n_tokens_per_item.max().item())

            sorted_indices_cpu = sorted_indices[:, :max_n_tokens].cpu()
            sorted_logprobs_cpu = sorted_log_probs[:, :max_n_tokens].cpu()
            n_tokens_cpu = n_tokens_per_item.cpu()

        for i in range(actual_batch_size):
            token_id = batch_token_ids[i]
            n_tokens = int(n_tokens_cpu[i].item())

            top_indices_i = sorted_indices_cpu[i, :n_tokens].tolist()
            top_logprobs_i = sorted_logprobs_cpu[i, :n_tokens].tolist()

            logprobs_dict = {str(ti): lp for ti, lp in zip(top_indices_i, top_logprobs_i)}
            results[str(token_id)] = logprobs_dict

    logger.info(f"Saving results to {output_file}...")

    output_data = {
        "model_name": model_name,
        "vocab_size": vocab_size,
        "cumulative_prob_threshold": cumulative_prob_threshold,
        "max_logprobs": max_logprobs,
        "results": results,
    }

    with open(output_file, "wb") as f:
        f.write(orjson.dumps(output_data))
        f.write(b"\n")

    logger.info(f"Benchmark complete. Results saved to {output_file}")
    return output_file


def load_benchmark(model_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> dict:
    """Load benchmark data from disk."""
    path = _benchmark_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark not found: {path}")
    with open(path, "rb") as f:
        return orjson.loads(f.read())["results"]


def _compute_top2_diff(logprobs_dict: dict[str, float]) -> float | None:
    """Compute difference between top 2 logprobs, or None if < 2 available."""
    if len(logprobs_dict) < 2:
        return None
    sorted_logprobs = sorted(logprobs_dict.values(), reverse=True)
    return sorted_logprobs[0] - sorted_logprobs[1]


def _get_ground_truth_ranking(benchmark: dict, delta: float) -> list[tuple[str, float]]:
    """Rank tokens by distance of their top-2 logprob difference to delta."""
    distances = []
    for token_id, logprobs_dict in benchmark.items():
        diff = _compute_top2_diff(logprobs_dict)
        if diff is not None:
            distances.append((token_id, abs(diff - delta)))
    distances.sort(key=lambda x: x[1])
    return distances


class InputSampler:
    """Provides lazy-cached sampling interface for a single input token."""

    def __init__(self, logprobs_dict: dict[str, float], rng: torch.Generator):
        self._logprobs_dict = logprobs_dict
        self._rng = rng
        self._samples_cache: dict[float, list[str]] = {}
        self._token_ids = list(logprobs_dict.keys())
        self._base_logprobs = torch.tensor([logprobs_dict[t] for t in self._token_ids])

    def sample(self, n: int, temperature: float = 1.0) -> list[str]:
        """Get n samples, using cache when possible."""
        cache = self._samples_cache.get(temperature, [])
        if len(cache) >= n:
            return cache[:n]

        logprobs = self._base_logprobs
        if temperature != 1.0:
            logprobs = logprobs / temperature

        probs = torch.softmax(logprobs, dim=0)
        indices = torch.multinomial(probs, n, replacement=True, generator=self._rng)
        samples = [self._token_ids[i] for i in indices.tolist()]

        self._samples_cache[temperature] = samples

        return samples

    def extend_samples(self, additional: int, temperature: float = 1.0) -> list[str]:
        """Extend the sample cache and return all samples."""
        cache = self._samples_cache.get(temperature, [])

        logprobs = self._base_logprobs
        if temperature != 1.0:
            logprobs = logprobs / temperature

        probs = torch.softmax(logprobs, dim=0)
        indices = torch.multinomial(probs, additional, replacement=True, generator=self._rng)
        new_samples = [self._token_ids[i] for i in indices.tolist()]
        cache.extend(new_samples)
        self._samples_cache[temperature] = cache
        return cache

    @property
    def total_sampled(self) -> int:
        return sum(len(cache) for cache in self._samples_cache.values())


class BenchmarkEvaluator:
    """Evaluates black-box methods for finding inputs with specific logprob characteristics."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        delta: float = 0.0,
    ):
        self.benchmark = load_benchmark(model_name)
        self.delta = delta

        # Filter out tokens with < 2 logprobs
        self._valid_results = {k: v for k, v in self.benchmark.items() if len(v) >= 2}
        self._available_tokens = list(self._valid_results.keys())

        self.ground_truth = _get_ground_truth_ranking(self._valid_results, delta)
        self._token_to_distance = {t[0]: t[1] for t in self.ground_truth}

    def get_available_tokens(self) -> list[str]:
        return self._available_tokens

    def create_sampler(self, token_id: str, rng: torch.Generator) -> InputSampler | None:
        """Create a sampler for the given input token."""
        logprobs_dict = self._valid_results.get(token_id)
        if logprobs_dict is None:
            return None
        return InputSampler(logprobs_dict, rng)

    def evaluate(
        self,
        method: Callable[["BenchmarkEvaluator", int, int, int], list[str]],
        query_budget: int,
        top_n: int = 10,
        seed: int = 42,
    ) -> dict:
        """Evaluate a method. Returns dict with score, overlap, and other metrics."""
        found_tokens = method(self, query_budget, top_n, seed)

        real_best_tokens = [t[0] for t in self.ground_truth[:top_n]]
        real_best_distances = [t[1] for t in self.ground_truth[:top_n]]
        avg_real_distance = sum(real_best_distances) / len(real_best_distances)

        found_distances = [self._token_to_distance.get(t, float("inf")) for t in found_tokens]
        avg_found_distance = (
            sum(found_distances) / len(found_distances) if found_distances else float("inf")
        )

        score = abs(avg_found_distance - avg_real_distance)
        overlap = len(set(found_tokens) & set(real_best_tokens))

        return {
            "score": score,
            "overlap": overlap,
            "top_n": top_n,
            "query_budget": query_budget,
            "avg_real_distance": avg_real_distance,
            "avg_found_distance": avg_found_distance,
            "found_tokens": found_tokens,
            "real_best_tokens": real_best_tokens,
        }


NAIVE_TEMPERATURES = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
NAIVE_SAMPLES_PER_TOKEN = [5, 10, 20, 30, 60, 100, 200]


def make_naive_method(
    temperature: float = 1.0,
    samples_per_token: int = 30,
) -> Callable[["BenchmarkEvaluator", int, int, int], list[str]]:
    """Create a naive method with specific temperature and samples_per_token."""

    def naive_method(
        evaluator: BenchmarkEvaluator,
        query_budget: int,
        top_n: int,
        seed: int,
    ) -> list[str]:
        """Sample each token, estimate top-2 diff from counts."""
        rng = torch.Generator()
        rng.manual_seed(seed)

        available_tokens = evaluator.get_available_tokens()
        spt = samples_per_token

        n_testable = query_budget // spt
        if n_testable == 0:
            spt = max(10, query_budget // min(len(available_tokens), top_n * 10))
            n_testable = query_budget // spt

        n_testable = min(n_testable, len(available_tokens))

        perm = torch.randperm(len(available_tokens), generator=rng)
        tokens_to_test = [available_tokens[i] for i in perm[:n_testable].tolist()]

        estimates = []
        for token_id in tokens_to_test:
            sampler = evaluator.create_sampler(token_id, rng)
            if sampler is None:
                continue

            samples = sampler.sample(spt, temperature=temperature)
            counts = Counter(samples)
            sorted_counts = sorted(counts.values(), reverse=True)

            if len(sorted_counts) < 2:
                estimated_diff = float("inf")
            else:
                total = sum(sorted_counts) + len(sorted_counts)
                p1 = (sorted_counts[0] + 1) / total
                p2 = (sorted_counts[1] + 1) / total
                # At temperature T, observed probs are p^(1/T), so multiply by T to recover original
                estimated_diff = temperature * (math.log(p1) - math.log(p2))

            distance = abs(estimated_diff - evaluator.delta)
            estimates.append((token_id, distance))

        estimates.sort(key=lambda x: x[1])
        chosen = [t[0] for t in estimates[:top_n]]
        if len(chosen) < top_n:
            chosen_set = set(chosen)
            # Fill the remainder with random tokens not already chosen
            remaining = [tok for tok in evaluator.get_available_tokens() if tok not in chosen_set]
            perm = torch.randperm(len(remaining), generator=rng)
            fill = [remaining[i] for i in perm[: (top_n - len(chosen))].tolist()]
            chosen += fill
        return chosen[:top_n]

    return naive_method


def random_baseline_method(
    evaluator: BenchmarkEvaluator,
    query_budget: int,
    top_n: int,
    seed: int,
) -> list[str]:
    """Random selection baseline."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    available_tokens = evaluator.get_available_tokens()
    perm = torch.randperm(len(available_tokens), generator=rng)
    return [available_tokens[i] for i in perm[:top_n].tolist()]


def evaluate(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    budgets: list[int] | None = None,
    top_n: int = 50,
):
    logging.basicConfig(level=logging.INFO)
    deltas = [0.0, 0.05, 0.1, 0.2, 0.4]

    if budgets is None:
        budgets = [int(b) for b in [1e5]]

    # Create output directory for heatmaps
    heatmap_dir = config.plots_dir / "best_tokens_benchmark"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # Collect all data first for combined plot
    all_data: dict[float, dict] = {}

    for delta in deltas:
        print("\n==============================")
        print(f"Evaluating for delta = {delta}")
        print("==============================")
        evaluator = BenchmarkEvaluator(model_name, delta)

        for budget in budgets:
            print(f"\n=== Query Budget: {budget} ===")

            result_random = evaluator.evaluate(random_baseline_method, budget, top_n=top_n)
            random_score = result_random["score"]
            print(f"Random:  score={random_score:.4f}, overlap={result_random['overlap']}/{top_n}")

            # Collect scores for heatmap
            scores = np.zeros((len(NAIVE_SAMPLES_PER_TOKEN), len(NAIVE_TEMPERATURES)))

            for i, temp in enumerate(NAIVE_TEMPERATURES):
                for j, spt in enumerate(NAIVE_SAMPLES_PER_TOKEN):
                    method = make_naive_method(temperature=temp, samples_per_token=spt)
                    result = evaluator.evaluate(method, budget, top_n=top_n)
                    scores[j, i] = result["score"]
                    print(
                        f"Naive(T={temp}, spt={spt}):  "
                        f"score={result['score']:.4f}, overlap={result['overlap']}/{top_n}"
                    )

            all_data[delta] = {"scores": scores, "random_score": random_score}

    # Create combined figure with subplots
    n_deltas = len(deltas)
    n_cols = math.ceil(math.sqrt(n_deltas))
    n_rows = math.ceil(n_deltas / n_cols)

    temp_labels = [f"{t:.0e}" if t < 0.1 else str(t) for t in NAIVE_TEMPERATURES]
    spt_labels = [str(spt) for spt in NAIVE_SAMPLES_PER_TOKEN]

    subplot_titles = [f"δ = {delta}" for delta in deltas]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Find global max for consistent colorscale
    global_max = max(data["random_score"] for data in all_data.values())

    for idx, delta in enumerate(deltas):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        data = all_data[delta]

        fig.add_trace(
            go.Heatmap(
                z=data["scores"],
                x=temp_labels,
                y=spt_labels,
                colorscale=[[0, "blue"], [1, "red"]],
                zmin=0,
                zmax=global_max,
                text=np.round(data["scores"], 3),
                texttemplate="%{text}",
                textfont={"size": 8},
                hovertemplate=f"δ={delta}<br>Temperature: %{{x}}<br>Samples/Token: %{{y}}<br>Score: %{{z:.4f}}<extra></extra>",
                showscale=(idx == 0),  # Only show colorbar for first subplot
                colorbar={"title": "Score"} if idx == 0 else None,
            ),
            row=row,
            col=col,
        )

        # Update axes labels
        fig.update_xaxes(title_text="Temperature" if row == n_rows else "", row=row, col=col)
        fig.update_yaxes(title_text="Samples/Token" if col == 1 else "", row=row, col=col)

    fig.update_layout(
        title=f"Score Heatmaps by Delta (budget={budgets[0]})<br><sub>Blue=0, Red=random score ({global_max:.4f})</sub>",
        width=350 * n_cols,
        height=300 * n_rows + 80,
    )

    # Save the combined heatmap
    filename = f"combined_budget_{budgets[0]}.png"
    filepath = heatmap_dir / filename
    fig.write_image(str(filepath), scale=2)
    print(f"\nSaved combined heatmap to {filepath}")


if __name__ == "__main__":
    fire.Fire({"create": create_benchmark, "evaluate": evaluate})
