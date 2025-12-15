"""
Benchmark to extract top logprobs for all single-token prompts.

For each possible 1-token input prompt (using the model's vocabulary),
run a forward pass with HuggingFace transformers and extract the top logprobs
such that their cumulative probability at T=1 exceeds a threshold (default 0.995).
"""

import logging
from pathlib import Path

import orjson
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from track_llm_apis.config import config
from track_llm_apis.util import slugify

logger = logging.getLogger("track-llm-apis")


def create_benchmark(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    cumulative_prob_threshold: float = 0.995,
    batch_size: int = 512,
    max_logprobs: int = 100,
) -> Path:
    """
    For each possible 1-token input prompt, run a forward pass to extract
    top logprobs until cumulative probability exceeds the threshold.

    Args:
        model_name: HuggingFace model name to benchmark
        cumulative_prob_threshold: Stop adding tokens when cumulative prob exceeds this
        batch_size: Number of prompts to process per batch

    Returns:
        Path to the output JSON file
    """
    output_dir = config.data_dir / "best_tokens_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{slugify(model_name)}.json"

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
            # logits shape: (batch_size, seq_len=1, vocab_size)
            logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
            logprobs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            # Get top-k by probability
            k = min(max_logprobs, probs.shape[-1])
            top_probs, top_indices = torch.topk(probs, k=k, dim=-1, sorted=False)
            top_logprobs = torch.gather(logprobs, dim=-1, index=top_indices)

            # Sort the top-k by probability
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

            top_indices = sorted_indices_cpu[i, :n_tokens].tolist()
            top_logprobs = sorted_logprobs_cpu[i, :n_tokens].tolist()

            logprobs_dict = {str(ti): lp for ti, lp in zip(top_indices, top_logprobs)}

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


if __name__ == "__main__":
    create_benchmark()
