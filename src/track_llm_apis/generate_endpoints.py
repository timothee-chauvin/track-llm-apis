import asyncio
from decimal import Decimal

import aiohttp
import requests

from track_llm_apis.config import Config
from track_llm_apis.main import Endpoint, OpenRouterClient
from track_llm_apis.util import gather_with_concurrency_streaming

logger = Config.logger


async def fetch_model_endpoints(session, model_id):
    """Fetch endpoints for a model and return those that claim to support logprobs"""
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    try:
        async with session.get(url) as response:
            data = await response.json()
            endpoints = data["data"]["endpoints"]

            model_endpoints_with_logprobs = []

            for endpoint in endpoints:
                # Only consider endpoints that claim to support logprobs
                if (
                    "logprobs" in endpoint["supported_parameters"]
                    and "top_logprobs" in endpoint["supported_parameters"]
                ):
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
                    model_endpoints_with_logprobs.append(endpoint_data)

            return model_endpoints_with_logprobs
    except Exception as e:
        logger.error(f"Error fetching endpoints for {model_id}: {e}")
        return []


async def test_endpoint_logprobs(endpoint):
    """Test if an endpoint actually returns logprobs when queried with 'x'"""
    client = OpenRouterClient()
    try:
        logger.info(f"Testing logprobs for {endpoint}...")
        response = await client.query(endpoint, "x")

        if response.error:
            return endpoint, False, response.error
        elif len(response.logprobs) == endpoint.get_max_logprobs():
            return endpoint, True, None
        else:
            return (
                endpoint,
                False,
                f"Expected {endpoint.get_max_logprobs()} logprobs, got {len(response.logprobs)}",
            )
    except Exception as e:
        return endpoint, False, str(e)


async def main():
    # Get the model IDs
    logger.info("Fetching model list...")
    response = requests.get("https://openrouter.ai/api/v1/models")
    model_ids = [model["id"] for model in response.json()["data"]]
    logger.info(f"Found {len(model_ids)} models")

    # Fetch endpoints that claim to support logprobs
    logger.info("Fetching endpoints that claim to support logprobs...")
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_model_endpoints(session, model_id) for model_id in model_ids]
        results = []
        total = len(tasks)
        async for result in gather_with_concurrency_streaming(20, *tasks):
            results.append(result)
            logger.info(f"Fetched endpoints: {len(results)}/{total}")

    # Flatten the results
    endpoints_with_logprobs = []
    for endpoint_list in results:
        endpoints_with_logprobs.extend(endpoint_list)

    logger.info(f"Found {len(endpoints_with_logprobs)} endpoints claiming to support logprobs")

    # Avoid getting ripped off by a $1T/Mtok endpoint: filter out any endpoints that cost too much per input or output token.
    max_cost = 200  # per million tokens
    endpoints_cost_ok = []
    for endpoint in endpoints_with_logprobs:
        if endpoint.cost[0] < max_cost and endpoint.cost[1] < max_cost:
            endpoints_cost_ok.append(endpoint)
        else:
            logger.info(
                f"Filtered out {endpoint.name} because it costs more than ${max_cost}/Mtok input or output: {endpoint.cost}"
            )

    endpoints_with_logprobs = endpoints_cost_ok

    # Test each endpoint to see if it actually returns logprobs
    logger.info("Testing endpoints for actual logprobs functionality...")
    test_tasks = [test_endpoint_logprobs(endpoint) for endpoint in endpoints_with_logprobs]
    test_results = []
    total_tests = len(test_tasks)
    async for result in gather_with_concurrency_streaming(20, *test_tasks):
        test_results.append(result)
        logger.info(f"Tested endpoints: {len(test_results)}/{total_tests}")

    # Separate successful and failed endpoints
    successful_endpoints = []
    failed_endpoints = []

    for endpoint, success, error in test_results:
        if success:
            successful_endpoints.append(endpoint)
        else:
            failed_endpoints.append((endpoint, error))

    # Output results
    print("#" * 100)
    print(
        f"SUCCESSFUL ENDPOINTS (returned the correct amount of logprobs with prompt 'x'): {len(successful_endpoints)}"
    )
    print("#" * 100)

    for endpoint in sorted(successful_endpoints, key=lambda x: x.cost[0]):
        print(
            f'    Endpoint("openrouter", "{endpoint.name}", "{endpoint.provider}", cost=({endpoint.cost[0]}, {endpoint.cost[1]})),'
        )

    print("\n" + "#" * 100)
    print(f"FAILED ENDPOINTS (with error messages): {len(failed_endpoints)}")
    print("#" * 100)

    for endpoint, error in failed_endpoints:
        print(f"❌ {endpoint.name} ({endpoint.provider}): {error}")

    print("\n" + "#" * 100)
    print("SUMMARY:")
    print(f"✅ Successful: {len(successful_endpoints)}")
    print(f"❌ Failed: {len(failed_endpoints)}")


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
