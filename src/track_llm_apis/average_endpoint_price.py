import asyncio
import logging
import statistics
from decimal import Decimal

import aiohttp
import fire
import requests

from track_llm_apis.config import config
from track_llm_apis.util import gather_with_concurrency_streaming

logger = config.logger
logger.setLevel(logging.DEBUG)


async def fetch_model_endpoints(session, model_id):
    """Fetch endpoints for a model and return (model_id, list of (provider_name, input_cost, output_cost) tuples)."""
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    try:
        async with session.get(url) as response:
            data = await response.json()
            endpoints = data["data"]["endpoints"]

            costs = []
            for endpoint in endpoints:
                provider_name = endpoint.get("provider_name", "unknown")
                input_cost = float((Decimal(endpoint["pricing"]["prompt"]) * 1_000_000).normalize())
                output_cost = float(
                    (Decimal(endpoint["pricing"]["completion"]) * 1_000_000).normalize()
                )
                costs.append((provider_name, input_cost, output_cost))

            return (model_id, costs)
    except Exception as e:
        logger.error(f"Error fetching endpoints for {model_id}: {e}")
        return (model_id, [])


async def main():
    # Get the model IDs
    logger.info("Fetching model list...")
    response = requests.get("https://openrouter.ai/api/v1/models")
    model_ids = [model["id"] for model in response.json()["data"]]
    logger.info(f"Found {len(model_ids)} models")

    # Fetch all endpoints
    logger.info("Fetching endpoints for all models...")
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_model_endpoints(session, model_id) for model_id in model_ids]
        results = []
        async for result in gather_with_concurrency_streaming(20, *tasks):
            results.append(result)

    # Flatten costs and track all endpoints with model info
    all_endpoints = []
    for model_id, costs in results:
        for provider_name, input_cost, output_cost in costs:
            combined = (input_cost + output_cost) / 2
            all_endpoints.append((model_id, provider_name, input_cost, output_cost, combined))

    logger.info(f"Total endpoints: {len(all_endpoints)}")

    # Log all endpoints sorted by increasing combined price
    all_endpoints.sort(key=lambda x: x[4])
    for model_id, provider_name, input_cost, output_cost, combined in all_endpoints:
        logger.debug(
            f"{model_id} ({provider_name}): (${input_cost:.2f}, ${output_cost:.2f}), avg ${combined:.2f}"
        )

    def print_stats(endpoints, label):
        """Print statistics for a list of (model_id, provider_name, input_cost, output_cost, combined) tuples."""
        input_costs = [e[2] for e in endpoints]
        output_costs = [e[3] for e in endpoints]
        combined_costs = [e[4] for e in endpoints]

        print(f"\n{'=' * 60}")
        print(f"{label} (per million tokens)")
        print(f"{'=' * 60}")

        print("\nINPUT COSTS:")
        print(f"  Average: ${statistics.mean(input_costs):.4f}")
        print(f"  Median:  ${statistics.median(input_costs):.4f}")
        print(f"  Min:     ${min(input_costs):.4f}")
        print(f"  Max:     ${max(input_costs):.4f}")
        print(f"  Std Dev: ${statistics.stdev(input_costs):.4f}")

        print("\nOUTPUT COSTS:")
        print(f"  Average: ${statistics.mean(output_costs):.4f}")
        print(f"  Median:  ${statistics.median(output_costs):.4f}")
        print(f"  Min:     ${min(output_costs):.4f}")
        print(f"  Max:     ${max(output_costs):.4f}")
        print(f"  Std Dev: ${statistics.stdev(output_costs):.4f}")

        print("\nCOMBINED (avg of input + output):")
        print(f"  Average: ${statistics.mean(combined_costs):.4f}")
        print(f"  Median:  ${statistics.median(combined_costs):.4f}")
        print(f"  Min:     ${min(combined_costs):.4f}")
        print(f"  Max:     ${max(combined_costs):.4f}")

        print(f"\nTotal endpoints: {len(endpoints)}")

    # Print stats for all endpoints
    print_stats(all_endpoints, f"ALL ENDPOINTS ({len(all_endpoints)} total)")

    # Exclude top 5% most expensive (by combined cost)
    sorted_by_combined = sorted(all_endpoints, key=lambda x: x[4])
    cutoff_idx = int(len(sorted_by_combined) * 0.95)
    filtered_endpoints = sorted_by_combined[:cutoff_idx]
    excluded_count = len(all_endpoints) - len(filtered_endpoints)

    print_stats(
        filtered_endpoints,
        f"EXCLUDING TOP 5% MOST EXPENSIVE ({excluded_count} endpoints removed)",
    )

    print(f"\n{'=' * 60}")
    print(f"Total models: {len(model_ids)}")
    print("=" * 60)


def entrypoint():
    asyncio.run(main())


if __name__ == "__main__":
    fire.Fire(entrypoint)
