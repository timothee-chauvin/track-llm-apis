import asyncio
import os
from decimal import Decimal

import aiohttp
import fire
import requests

from track_llm_apis.config import config
from track_llm_apis.main import Endpoint, OpenRouterClient
from track_llm_apis.util import gather_with_concurrency_streaming, retry_with_exponential_backoff

logger = config.logger


async def fetch_all_endpoints(session):
    """Fetch all endpoints from OpenRouter API"""
    logger.info("Fetching model list...")
    response = requests.get("https://openrouter.ai/api/v1/models")
    model_ids = [model["id"] for model in response.json()["data"]]
    logger.info(f"Found {len(model_ids)} models")

    async def fetch_model_endpoints(model_id):
        """Fetch endpoints for a single model"""
        url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
        try:
            async with session.get(url) as response:
                data = await response.json()
                endpoints = data["data"]["endpoints"]

                model_endpoints = []
                for endpoint in endpoints:
                    endpoint_data = Endpoint(
                        source="openrouter",
                        name=model_id,
                        provider=endpoint["tag"],
                        cost=(
                            float((Decimal(endpoint["pricing"]["prompt"]) * 1_000_000).normalize()),
                            float(
                                (Decimal(endpoint["pricing"]["completion"]) * 1_000_000).normalize()
                            ),
                        ),
                    )
                    model_endpoints.append(endpoint_data)
                return model_endpoints
        except Exception as e:
            logger.error(f"Error fetching endpoints for {model_id}: {e}")
            return []

    # Fetch all endpoints
    logger.info("Fetching all endpoints...")
    tasks = [fetch_model_endpoints(model_id) for model_id in model_ids]
    results = []
    async for result in gather_with_concurrency_streaming(20, *tasks):
        results.extend(result)
        logger.info(f"Fetched endpoints: {len(results)} total so far")

    logger.info(f"Found {len(results)} total endpoints")
    return results


async def test_endpoint_simple(endpoint):
    """Test endpoint with a simple request (no logprobs)"""

    async def _make_request(endpoint):
        try:
            # Create a simple request without logprobs
            request_data = {
                "model": endpoint.name,
                "messages": [{"role": "user", "content": "x"}],
                "max_completion_tokens": 1,
                "temperature": 1.0,
                "provider": {
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
            }
            if endpoint.provider:
                request_data["provider"]["only"] = [endpoint.provider]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
                    json=request_data,
                ) as resp:
                    if resp.ok:
                        response = await resp.json()
                        # Check if we got a valid response
                        if "choices" in response and response["choices"]:
                            return endpoint, True, None
                        else:
                            return endpoint, False, "No choices in response"
                    else:
                        error_text = await resp.text()
                        logger.error(
                            f"Error in response from {endpoint}: {resp.status} {error_text}"
                        )
                        return endpoint, False, f"HTTP {resp.status}: {error_text}"
        except Exception as e:
            logger.error(f"Error testing endpoint {endpoint}: {e}")
            return endpoint, False, str(e)

    return await retry_with_exponential_backoff(
        _make_request,
        endpoint,
        max_retries=config.api.max_retries,
    )


async def test_endpoint_with_logprobs(endpoint):
    """Test endpoint with logprobs enabled"""
    client = OpenRouterClient()
    try:
        logger.info(f"Testing logprobs for {endpoint}...")
        response = await client.query(endpoint, "x", temperature=1.0)

        if response.error:
            return endpoint, False, response.error
        elif response.logprobs:
            return endpoint, True, None
        else:
            return endpoint, False, "No logprobs returned"
    except Exception as e:
        return endpoint, False, str(e)


async def main():
    config.api.max_retries = 2

    # Step 1: Fetch all endpoints
    async with aiohttp.ClientSession() as session:
        all_endpoints = await fetch_all_endpoints(session)

    max_cost = 1000  # per million tokens
    all_endpoints = [e for e in all_endpoints if e.cost[0] < max_cost and e.cost[1] < max_cost]

    logger.info(f"Testing {len(all_endpoints)} endpoints")

    # Step 2: Test all endpoints with simple requests (no logprobs)
    logger.info("Testing all endpoints with simple requests (no logprobs)...")
    simple_test_tasks = [test_endpoint_simple(endpoint) for endpoint in all_endpoints]
    simple_results = []
    total_simple = len(simple_test_tasks)
    async for result in gather_with_concurrency_streaming(50, *simple_test_tasks):
        simple_results.append(result)
        logger.info(f"Simple tests completed: {len(simple_results)}/{total_simple}")

    working_endpoints = []
    failed_simple_endpoints = []

    for endpoint, success, error in simple_results:
        if success:
            working_endpoints.append(endpoint)
        else:
            failed_simple_endpoints.append((endpoint, error))

    logger.info(f"Endpoints that work with simple requests: {len(working_endpoints)}")
    logger.info(f"Endpoints that failed simple requests: {len(failed_simple_endpoints)}")

    # Step 3: Test working endpoints with logprobs
    logger.info("Testing working endpoints with logprobs...")
    logprob_test_tasks = [test_endpoint_with_logprobs(endpoint) for endpoint in working_endpoints]
    logprob_results = []
    total_logprob = len(logprob_test_tasks)
    async for result in gather_with_concurrency_streaming(50, *logprob_test_tasks):
        logprob_results.append(result)
        logger.info(f"Logprob tests completed: {len(logprob_results)}/{total_logprob}")

    logprob_supported = []
    logprob_not_supported = []

    for endpoint, success, error in logprob_results:
        if success:
            logprob_supported.append(endpoint)
        else:
            logprob_not_supported.append((endpoint, error))

    print("\n" + "=" * 80)
    print("LOGPROB PREVALENCE STATISTICS")
    print("=" * 80)

    total_tested = len(working_endpoints)
    total_all_endpoints = len(all_endpoints)

    print(f"\nTotal endpoints found: {total_all_endpoints}")
    print(f"Endpoints that respond to simple requests: {len(working_endpoints)}")
    print(f"Endpoints that failed simple requests: {len(failed_simple_endpoints)}")

    print("\nAmong working endpoints:")
    print(f"  Endpoints supporting logprobs: {len(logprob_supported)}")
    print(f"  Endpoints NOT supporting logprobs: {len(logprob_not_supported)}")

    working_fraction = len(working_endpoints) / total_all_endpoints
    print(
        f"Fraction of endpoints that work: {working_fraction:.3f} ({len(working_endpoints)}/{total_all_endpoints})"
    )
    logprob_fraction = len(logprob_supported) / total_tested
    print(
        f"\nFraction of working endpoints that support logprobs: {logprob_fraction:.3f} ({len(logprob_supported)}/{total_tested})"
    )

    print("\n" + "=" * 50)
    print("PROVIDER BREAKDOWN")
    print("=" * 50)

    print("All endpoints with logprobs:")
    for endpoint in logprob_supported:
        print(f"  {endpoint.name} ({endpoint.provider})")

    provider_stats = {}
    for endpoint in logprob_supported:
        provider = endpoint.provider.split("/")[0]  # remove dtype
        if provider not in provider_stats:
            provider_stats[provider] = {"supported": 0, "not_supported": 0}
        provider_stats[provider]["supported"] += 1

    for endpoint, error in logprob_not_supported:
        provider = endpoint.provider.split("/")[0]  # remove dtype
        if provider not in provider_stats:
            provider_stats[provider] = {"supported": 0, "not_supported": 0}
        provider_stats[provider]["not_supported"] += 1

    print("\nProviders with logprob support:")
    for provider, stats in sorted(provider_stats.items()):
        total = stats["supported"] + stats["not_supported"]
        if stats["supported"] > 0:
            fraction = stats["supported"] / total
            print(f"  {provider}: {stats['supported']}/{total} ({fraction:.3f})")

    print("\nProviders without logprob support:")
    for provider, stats in sorted(provider_stats.items()):
        if stats["supported"] == 0 and stats["not_supported"] > 0:
            print(f"  {provider}: 0/{stats['not_supported']} (0.000)")

    print("\n" + "=" * 50)
    print("MODELS NEVER SUPPORTING LOGPROBS")
    print("=" * 50)

    sometimes_supported_models = set()
    never_supported_models = set()

    for endpoint in logprob_supported:
        model_name = endpoint.name
        if model_name not in sometimes_supported_models:
            sometimes_supported_models.add(model_name)

    for endpoint, error in logprob_not_supported:
        model_name = endpoint.name
        if model_name not in sometimes_supported_models:
            never_supported_models.add(model_name)

    print(
        f"\nModels that never support logprobs across any provider ({len(never_supported_models)} total):"
    )
    for model_name in never_supported_models:
        print(f"  {model_name}")

    # Show some example errors
    print("\n" + "=" * 50)
    print("EXAMPLE ERRORS (first 10)")
    print("=" * 50)
    for i, (endpoint, error) in enumerate(logprob_not_supported[:10]):
        print(f"{i + 1}. {endpoint.name} ({endpoint.provider}): {error}")


def entrypoint():
    asyncio.run(main())


if __name__ == "__main__":
    fire.Fire(entrypoint)
