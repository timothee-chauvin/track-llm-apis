import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

from track_llm_apis.config import Config

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
