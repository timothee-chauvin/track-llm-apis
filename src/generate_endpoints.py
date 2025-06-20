import json

import requests

# Get the model IDs from https://openrouter.ai/api/v1/models
response = requests.get("https://openrouter.ai/api/v1/models")
model_ids = [model["id"] for model in response.json()["data"]]

print(model_ids)

endpoints_with_logprobs = []
endpoints_without_logprobs = []

# For each model ID, get its available providers
for model_id in model_ids:
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    response = requests.get(url)
    endpoints = response.json()["data"]["endpoints"]
    for endpoint in endpoints:
        if (
            "logprobs" in endpoint["supported_parameters"]
            and "top_logprobs" in endpoint["supported_parameters"]
        ):
            endpoints_with_logprobs.append(
                {
                    "model_id": model_id,
                    "provider": endpoint["tag"],
                    "cost": (endpoint["pricing"]["prompt"], endpoint["pricing"]["completion"]),
                }
            )
        else:
            endpoints_without_logprobs.append(
                {
                    "model_id": model_id,
                    "provider": endpoint["tag"],
                    "cost": (endpoint["pricing"]["prompt"], endpoint["pricing"]["completion"]),
                }
            )

print(f"Endpoints with logprobs (len={len(endpoints_with_logprobs)}):")
print(json.dumps(endpoints_with_logprobs, indent=2))

print(f"Endpoints without logprobs (len={len(endpoints_without_logprobs)}):")
print(json.dumps(endpoints_without_logprobs, indent=2))
