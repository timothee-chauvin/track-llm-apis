import asyncio
import json
from decimal import Decimal

import aiohttp
import fire
import requests

from track_llm_apis.config import config
from track_llm_apis.main import Endpoint, OpenRouterClient
from track_llm_apis.util import gather_with_concurrency_streaming

logger = config.logger


async def fetch_model_endpoints(session, model_id):
    """Fetch endpoints for a model and return a dict:
    {"logprobs": the endpoints that claim to support logprobs,
    "no_logprobs": the endpoints that claim not to support logprobs}
    """
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    try:
        async with session.get(url) as response:
            data = await response.json()
            endpoints = data["data"]["endpoints"]

            model_endpoints_with_logprobs = []
            model_endpoints_without_logprobs = []

            for endpoint in endpoints:
                endpoint_data = Endpoint(
                    source="openrouter",
                    name=model_id,
                    provider=endpoint["tag"],
                    cost=(
                        float((Decimal(endpoint["pricing"]["prompt"]) * 1_000_000).normalize()),
                        float((Decimal(endpoint["pricing"]["completion"]) * 1_000_000).normalize()),
                    ),
                )
                if (
                    "logprobs" in endpoint["supported_parameters"]
                    and "top_logprobs" in endpoint["supported_parameters"]
                ):
                    model_endpoints_with_logprobs.append(endpoint_data)
                else:
                    model_endpoints_without_logprobs.append(endpoint_data)

            return {
                "logprobs": model_endpoints_with_logprobs,
                "no_logprobs": model_endpoints_without_logprobs,
            }
    except Exception as e:
        logger.error(f"Error fetching endpoints for {model_id}: {e}")
        return {"logprobs": [], "no_logprobs": []}


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


async def main(add_to_existing: bool = False):
    config.max_retries = 5

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
    endpoints_claim_logprobs = []
    endpoints_claim_no_logprobs = []
    for result in results:
        endpoints_claim_logprobs.extend(result["logprobs"])
        endpoints_claim_no_logprobs.extend(result["no_logprobs"])

    logger.info(f"Found {len(endpoints_claim_logprobs)} endpoints claiming to support logprobs")
    logger.info(
        f"Found {len(endpoints_claim_no_logprobs)} endpoints claiming not to support logprobs"
    )

    # Avoid getting ripped off by a $1T/Mtok endpoint: filter out any endpoints that cost too much per input or output token.
    max_cost = 200  # per million tokens
    endpoints_cost_ok = []
    for endpoint in endpoints_claim_logprobs:
        if endpoint.cost[0] < max_cost and endpoint.cost[1] < max_cost:
            endpoints_cost_ok.append(endpoint)
        else:
            logger.info(
                f"Filtered out {endpoint.name} because it costs more than ${max_cost}/Mtok input or output: {endpoint.cost}"
            )

    endpoints_claim_logprobs = endpoints_cost_ok

    # Test each endpoint to see if it actually returns logprobs
    logger.info("Testing endpoints for actual logprobs functionality...")
    test_tasks = [test_endpoint_logprobs(endpoint) for endpoint in endpoints_claim_logprobs]
    test_results = []
    total_tests = len(test_tasks)
    async for result in gather_with_concurrency_streaming(50, *test_tasks):
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

    for endpoint in sorted(
        successful_endpoints, key=lambda x: (x.cost[0], x.cost[1], x.name, x.provider)
    ):
        print(
            f'    Endpoint("openrouter", "{endpoint.name}", provider="{endpoint.provider}", cost=({endpoint.cost[0]}, {endpoint.cost[1]})),'
        )

    print("\n" + "#" * 100)
    print(f"FAILED ENDPOINTS (with error messages): {len(failed_endpoints)}")
    print("#" * 100)

    for endpoint, error in failed_endpoints:
        print(f"❌ {endpoint.name} ({endpoint.provider}): {error}")

    # Analysis by provider
    provider_stats = {}
    for endpoint, success, error in test_results:
        provider = endpoint.provider
        provider = provider.split("/")[0]  # remove the dtype
        if provider not in provider_stats:
            provider_stats[provider] = {"successful": 0, "failed": 0, "errors": {}}
        if success:
            provider_stats[provider]["successful"] += 1
        else:
            provider_stats[provider]["failed"] += 1
            provider_stats[provider]["errors"][str(endpoint)] = error

    always_logprobs = []
    sometimes_logprobs = []
    sometimes_logprobs_errors = []
    never_logprobs = []
    never_logprobs_errors = []

    for provider, stats in sorted(provider_stats.items()):
        successful_count = stats["successful"]
        failed_count = stats["failed"]
        errors = stats["errors"]

        if failed_count == 0 and successful_count > 0:
            always_logprobs.append((provider, successful_count))
        elif successful_count > 0 and failed_count > 0:
            sometimes_logprobs.append((provider, successful_count, failed_count))
            sometimes_logprobs_errors.extend(errors)
        elif successful_count == 0 and failed_count > 0:
            never_logprobs.append((provider, failed_count))
            never_logprobs_errors.extend(errors)

    print("\n" + "#" * 100)
    print("PROVIDER SUMMARY")
    print("#" * 100)

    print(f"\nProviders always returning logprobs: {len(always_logprobs)}")
    n_endpoints_always = 0
    for provider, total in sorted(always_logprobs):
        n_endpoints_always += total
        print(f"  - {provider} ({total} endpoints)")
    print(f"Total endpoints: {n_endpoints_always}")

    n_endpoints_with = 0
    n_endpoints_without = 0
    print(f"\nProviders sometimes returning logprobs: {len(sometimes_logprobs)}")
    for provider, successful, failed in sorted(sometimes_logprobs):
        n_endpoints_with += successful
        n_endpoints_without += failed
        print(
            f"  - {provider} ({successful} with logprobs, {failed} without, total {successful + failed})"
        )
        print("Errors:")
        print(json.dumps(provider_stats[provider]["errors"], indent=2))
    print(f"Total endpoints: {n_endpoints_with} with, {n_endpoints_without} without")

    print(f"\nProviders never returning logprobs: {len(never_logprobs)}")
    n_endpoints_never = 0
    for provider, total in sorted(never_logprobs):
        n_endpoints_never += total
        print(f"  - {provider} ({total} endpoints)")
        print("Errors:")
        print(json.dumps(provider_stats[provider]["errors"], indent=2))
    print(f"Total endpoints: {n_endpoints_never}")

    providers_claim_no_logprobs = set(
        [endpoint.provider.split("/")[0] for endpoint in endpoints_claim_no_logprobs]
    )
    n_providers_claim_no_logprobs = len(providers_claim_no_logprobs)
    print(f"\nProviders claiming not to support logprobs: {n_providers_claim_no_logprobs}")
    for provider in sorted(providers_claim_no_logprobs):
        print(
            f"  - {provider} ({len([endpoint for endpoint in endpoints_claim_no_logprobs if endpoint.provider.split('/')[0] == provider])} endpoints)"
        )
    print(f"Total endpoints: {len(endpoints_claim_no_logprobs)}")

    print("\n" + "#" * 100)
    print("SUMMARY:")
    total_endpoints = len(endpoints_claim_logprobs) + len(endpoints_claim_no_logprobs)
    print(
        f"Endpoints claiming not to support logprobs: {len(endpoints_claim_no_logprobs)} / {total_endpoints} ({len(endpoints_claim_no_logprobs) / total_endpoints * 100:.2f}%)"
    )
    print(
        f"Endpoints claiming to return logprobs but didn't: {len(failed_endpoints)} / {total_endpoints} ({len(failed_endpoints) / total_endpoints * 100:.2f}%)"
    )
    print(
        f"Endpoints claiming to support logprobs and did: {len(successful_endpoints)} / {total_endpoints} ({len(successful_endpoints) / total_endpoints * 100:.2f}%)"
    )

    if add_to_existing:
        from track_llm_apis.main import ENDPOINTS_EXTENDED

        all_endpoints = successful_endpoints
        for e in ENDPOINTS_EXTENDED:
            if e.name in ["x-ai/grok-3-beta", "openai/gpt-4-1106-preview"]:
                print()
            if e.source != "openrouter":
                continue
            # Don't add already existing endpoints, taking into account that the price may have changed
            if any(
                e.name == existing_e.name and e.provider == existing_e.provider
                for existing_e in all_endpoints
            ):
                continue
            all_endpoints.append(e)

        print("\n" + "#" * 100)
        print("SUCCESSFUL OR EXISTING ENDPOINTS:")
        for endpoint in sorted(
            all_endpoints, key=lambda x: (x.cost[0], x.cost[1], x.name, x.provider)
        ):
            print(
                f'    Endpoint("openrouter", "{endpoint.name}", provider="{endpoint.provider}", cost=({endpoint.cost[0]}, {endpoint.cost[1]})),'
            )
        print(f"Total successful or existing endpoints: {len(all_endpoints)}")


def entrypoint(add_to_existing: bool = False):
    add_to_existing = True
    asyncio.run(main(add_to_existing=add_to_existing))


# Run the async main function
if __name__ == "__main__":
    fire.Fire(entrypoint)
