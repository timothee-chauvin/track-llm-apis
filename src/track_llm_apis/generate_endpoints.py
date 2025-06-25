import asyncio
import json
from decimal import Decimal

import aiohttp
import requests

from track_llm_apis.config import Config
from track_llm_apis.util import gather_with_concurrency

logger = Config.logger

# Get the model IDs
response = requests.get("https://openrouter.ai/api/v1/models")
model_ids = [model["id"] for model in response.json()["data"]]

print(model_ids)

endpoints_with_logprobs = []
endpoints_without_logprobs = []


async def fetch_model_endpoints(session, model_id):
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    try:
        logger.info(f"Fetching endpoints for {model_id}...")
        async with session.get(url) as response:
            data = await response.json()
            endpoints = data["data"]["endpoints"]

            model_endpoints_with_logprobs = []
            model_endpoints_without_logprobs = []

            for endpoint in endpoints:
                endpoint_data = {
                    "model_id": model_id,
                    "provider": endpoint["tag"],
                    "cost": (  # the API returns prices per token, we want per million tokens
                        f"{(Decimal(endpoint['pricing']['prompt']) * 1_000_000).normalize():f}",
                        f"{(Decimal(endpoint['pricing']['completion']) * 1_000_000).normalize():f}",
                    ),
                }

                if (
                    "logprobs" in endpoint["supported_parameters"]
                    and "top_logprobs" in endpoint["supported_parameters"]
                ):
                    model_endpoints_with_logprobs.append(endpoint_data)
                else:
                    model_endpoints_without_logprobs.append(endpoint_data)

            return model_endpoints_with_logprobs, model_endpoints_without_logprobs
    except Exception as e:
        print(f"Error fetching endpoints for {model_id}: {e}")
        return [], []


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_model_endpoints(session, model_id) for model_id in model_ids]
        results = await gather_with_concurrency(20, *tasks)

    # Combine results
    for with_logprobs, without_logprobs in results:
        endpoints_with_logprobs.extend(with_logprobs)
        endpoints_without_logprobs.extend(without_logprobs)


# Run the async main function
asyncio.run(main())

print("#" * 100)
print(f"Endpoints with logprobs (len={len(endpoints_with_logprobs)}):")
print(json.dumps(endpoints_with_logprobs, indent=2))

print("#" * 100)
print(f"Endpoints without logprobs (len={len(endpoints_without_logprobs)}):")
print(json.dumps(endpoints_without_logprobs, indent=2))

print("#" * 100)
print("Endpoints with logprobs in usable format:")
for endpoint in sorted(endpoints_with_logprobs, key=lambda x: float(x["cost"][0])):
    print(
        f'    Endpoint("openrouter", "{endpoint["model_id"]}", "{endpoint["provider"]}", cost=({endpoint["cost"][0]}, {endpoint["cost"][1]})),'
    )
