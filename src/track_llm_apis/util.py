import asyncio
import contextlib
import gc
import hashlib
import os
import random
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import torch
import xxhash
from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv

from track_llm_apis.config import Config

load_dotenv()

logger = Config.logger


async def gather_with_concurrency(n, *coros):
    # Taken from https://stackoverflow.com/a/61478547
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


async def gather_with_concurrency_streaming(n, *coros):
    """Version that yields results as they complete"""
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    # Create the semaphore-wrapped coroutines
    sem_coros = [sem_coro(c) for c in coros]

    # Use as_completed to get results as they finish
    for coro in asyncio.as_completed(sem_coros):
        yield await coro


def trim_to_length(s: str, length: int) -> str:
    return s[:length] + "..." if len(s) > length else s


async def retry_with_exponential_backoff(
    func: Callable[..., Awaitable[Any]],
    *args,
    max_retries: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (aiohttp.ClientError, asyncio.TimeoutError),
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504),
    **kwargs,
) -> Any:
    """
    Retry an async function with exponential backoff.

    Args:
        func: The async function to retry
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay between retries
        jitter: Whether to add random jitter to delay
        retryable_exceptions: Tuple of exception types that should trigger retries
        retryable_status_codes: HTTP status codes that should trigger retries
        **kwargs: Keyword arguments for the function
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            return result
        except aiohttp.ClientResponseError as e:
            # Check if this is a retryable HTTP status code
            if e.status in retryable_status_codes and attempt < max_retries:
                wait_time = min(max_delay, (base_delay * (2**attempt)))
                if jitter:
                    wait_time *= random.uniform(0.9, 1.1)

                logger.warning(
                    f"HTTP {e.status} error. Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries + 1})"
                )
                await asyncio.sleep(wait_time)
                last_exception = e
                continue
            else:
                raise e
        except retryable_exceptions as e:
            if attempt < max_retries:
                wait_time = min(max_delay, (base_delay * (2**attempt)))
                if jitter:
                    wait_time *= random.uniform(0.9, 1.1)

                logger.warning(
                    f"Retryable error: {e}. Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries + 1})"
                )
                await asyncio.sleep(wait_time)
                last_exception = e
                continue
            else:
                raise e
        except Exception as e:
            # Non-retryable exceptions are re-raised immediately
            logger.error(f"Non-retryable error: {e}")
            raise e

    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected end of retry loop")


def load_lmsys_chat_1m(
    gpt4_filter: bool = True,
    redacted_filter: bool = True,
    flagged_filter: bool = True,
    first_turn_only: bool = True,
    use_cache: bool = True,
) -> Dataset:
    """
    Load the LMSYS Chat 1M dataset, returning only the "conversation" column.

    Args:
        gpt4_filter: Filter out non-GPT-4 conversations
        redacted_filter: Filter out redacted conversations
        flagged_filter: Filter out conversations with at least one message flagged per the "openai_moderation" column
        first_turn_only: Only keep the first turn of each conversation
        use_cache: Use the processed dataset if it exists on disk, otherwise create it
    """
    ds_name = "lmsys/lmsys-chat-1m"
    logger.info(f"Loading the {ds_name} dataset...")
    cache_path = Config.datasets_dir / f"{slugify(ds_name, hash_length=0)}"
    if use_cache:
        if cache_path.exists():
            logger.info(f"Already processed dataset found at {cache_path}, loading...")
            dataset = load_from_disk(str(cache_path))
            assert isinstance(dataset, Dataset)
            return dataset
        logger.info(f"No processed dataset found at {cache_path}, creating...")

    def filter_fn(model, redacted):
        if gpt4_filter and not redacted_filter:
            return model == "gpt-4"
        elif not gpt4_filter and redacted_filter:
            return ~redacted
        elif gpt4_filter and redacted_filter:
            return (model == "gpt-4") & (~redacted)
        else:
            return True

    def flagged_filter_fn(moderation):
        return all(not m["flagged"] for m in moderation)

    dataset = load_dataset("lmsys/lmsys-chat-1m", token=os.getenv("HF_TOKEN"), split="train")
    assert isinstance(dataset, Dataset)

    logger.info("Filtering dataset...")
    dataset = (
        dataset.with_format("np")
        .filter(
            filter_fn,
            input_columns=["model", "redacted"],
            batched=True,
        )
        .with_format(None)
    )
    if first_turn_only:
        dataset = dataset.map(lambda x: {"conversation": x["conversation"][:2]}, batched=False)
    if flagged_filter:
        dataset = dataset.filter(
            flagged_filter_fn, input_columns=["openai_moderation"], batched=False
        )
    assert all(s["conversation"][0]["role"] == "user" for s in dataset)  # pyright: ignore[reportArgumentType,reportCallIssue]
    assert all(s["conversation"][1]["role"] == "assistant" for s in dataset)  # pyright: ignore[reportArgumentType,reportCallIssue]
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "conversation"])
    if use_cache:
        dataset.save_to_disk(str(cache_path))
    return dataset


def slugify(s: str, max_length: int = 50, hash_length: int = 8) -> str:
    """
    Convert a string to a slugified version suitable for Linux filenames.

    Special characters are hex-encoded to preserve information while keeping
    the filename safe. For example, "|" becomes "-x7c-".

    Args:
        s: The input string to slugify
        max_length: Maximum length of the output without the hash (default: 50)
        hash_length: Length of the hash to append to the output (default: 8)

    Returns:
        A slugified string safe for use as a Linux filename
    """
    slug = ""

    for char in s:
        if char.isalnum() or char in "._-":
            slug += char
        elif char == " ":
            slug += "-"
        else:
            slug += f"-x{ord(char):02x}-"

    slug = slug[:max_length]

    if hash_length > 0:
        string_hash = hashlib.md5(s.encode("utf-8")).hexdigest()[:hash_length]
        slug += "_" + string_hash

    return slug


def available_gpu_memory_fraction():
    """
    Calculate the fraction of GPU memory that is currently available.
    """
    free, total = torch.cuda.mem_get_info()
    return free / total


def used_gpu_memory(cleanup: bool = False, as_str: bool = False) -> float | str:
    if cleanup:
        gc.collect()
        torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    if as_str:
        return f"Used GPU memory: {(total - free) / 1024**3:.2f} GB / {total / 1024**3:.2f} GB"
    else:
        return total - free


def format_mmlu_prompt(mmlu_item: dict) -> str:
    a, b, c, d = mmlu_item["choices"]
    choices_str = f"A. {a}\nB. {b}\nC. {c}\nD. {d}"
    return f"Answer the following multiple choice question. The entire content of your response should be of the following format: ‘ANSWER: $LETTER’ (without quotes) where LETTER is one of A,B,C,D.\n\n{mmlu_item['question']}\n\n{choices_str}"


def format_wikipedia_prompt(wikipedia_item: dict) -> str:
    """Copied from https://github.com/i-gao/model-equality-testing/blob/fd2ee24d75c9fef87debff8caefa0c04d4a5d374/experiments/prompts.py"""
    out = "Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.\n"
    out += '"' + wikipedia_item["text"][:100] + '..."'
    return out


def get_model_hash(model):
    """
    Compute a hash of the model's parameters.

    Args:
        model: PyTorch model

    Returns:
        str: Hexadecimal hash string representing the model state
    """
    hasher = xxhash.xxh64()

    # Parameters
    for _, param in sorted(model.named_parameters()):
        # Convert to float32 before converting to bytes to ensure consistent hashing
        param_data = param.detach().cpu().to(torch.float32).numpy().tobytes()
        hasher.update(param_data)

    # Buffers
    for _, buffer in sorted(model.named_buffers()):
        if buffer is not None:
            buffer_data = buffer.detach().cpu().to(torch.float32).numpy().tobytes()
            hasher.update(buffer_data)

    return hasher.hexdigest()


def get_dataset_hash(dataset: Dataset) -> str:
    """
    Compute a hash of the dataset.
    """
    hasher = xxhash.xxh64()
    for item in dataset:
        hasher.update(str(item).encode("utf-8"))
    return hasher.hexdigest()


def fast_hash(s: str) -> str:
    return xxhash.xxh64(s).hexdigest()


@contextlib.contextmanager
def temporary_env(variable_name: str, value: str):
    """Context manager for temporarily setting an environment variable."""
    original_value = os.getenv(variable_name)
    os.environ[variable_name] = value
    try:
        yield
    finally:
        if original_value is None:
            os.environ.pop(variable_name, None)
        else:
            os.environ[variable_name] = original_value


def copy_model_to(model, device: str, dtype: torch.dtype | None = torch.bfloat16):
    """Copy a model to a new device, without first creating a copy on the original device."""
    logger.info(f"Copying model to {device} with dtype {dtype}...")
    new_model = type(model)(model.config)
    if dtype is not None:
        new_model.to(device, dtype=dtype)
    else:
        new_model.to(device)
    new_model.load_state_dict(model.state_dict())
    return new_model


def patch_chat_template(tokenizer):
    chat_template = tokenizer.chat_template
    if "{% generation %}" in chat_template:
        return
    else:
        h = fast_hash(chat_template)
        if h in Config.chat_templates:
            tokenizer.chat_template = Config.chat_templates[h]["template"]
        else:
            raise ValueError(
                f"Chat template hash {h} not found in Config.chat_templates. You may need to update the chat_templates.toml file."
            )
    return
