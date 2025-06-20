import asyncio


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
