import json

from datasets import load_dataset

from track_llm_apis.config import config

OUTPUT_FILE = config.baselines_dir / "gao2025" / "wikipedia_first_100.json"


def download_wikipedia_first_100():
    """
    Download and save the first 100 Wikipedia articles for each language from Gao 2025.
    """
    wikipedia = {}
    for language in ["en", "de", "es", "fr", "ru"]:
        wikipedia_stream = load_dataset(
            "Cohere/wikipedia-2023-11-embed-multilingual-v3",
            language,
            split="train",
            streaming=True,
        )
        first_100 = wikipedia_stream.take(100)  # pyright: ignore[reportAttributeAccessIssue]
        first_100 = first_100.remove_columns(["url", "_id", "title", "emb"])  # pyright: ignore[reportAttributeAccessIssue]
        first_100 = list(first_100)
        wikipedia[f"wikipedia_{language}"] = first_100

    with open(OUTPUT_FILE, "w") as f:
        json.dump(wikipedia, f)


def get_wikipedia_samples(n: int, seed: int) -> list[dict[str, str]]:
    """
    Return a sample of n Wikipedia articles across all languages, extracted from the Gao 2025 data.

    The wikipedia items are downloaded and cached.
    """
    if not OUTPUT_FILE.exists():
        download_wikipedia_first_100()
    with open(OUTPUT_FILE) as f:
        wikipedia_first_100 = json.load(f)

    with open(
        config.baselines_dir / "gao2025" / "wikipedia_prompt_indices_test" / f"{n}" / f"{seed}.json"
    ) as f:
        ids_by_language = json.load(f)

    wikipedia = []
    for language in ["en", "de", "es", "fr", "ru"]:
        for i in ids_by_language[f"wikipedia_{language}"]:
            wikipedia.append(wikipedia_first_100[f"wikipedia_{language}"][i])
    return wikipedia


if __name__ == "__main__":
    download_wikipedia_first_100()
