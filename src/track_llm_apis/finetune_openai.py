import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

from track_llm_apis.config import config
from track_llm_apis.util import load_lmsys_chat_1m

random.seed(config.seed)
np.random.seed(config.seed)

logger = config.logger

load_dotenv()

client = OpenAI()


MODELS = [
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
]
LR_MULTIPLIERS = [1e-4, 1e-2, 1]
SAMPLE_SIZES = [10, 20, 40, 80]
EPOCHS = 1
BATCH_SIZE = 1
DATASET_NAME = "lmsys-chat-1m"


def file_name(sample_size: int) -> str:
    return f"{DATASET_NAME}_{sample_size}_seed={config.seed}.jsonl"


def generate_jsonl(dataset: Dataset) -> Path:
    """
    Generate a JSONL file for OpenAI from `dataset`, save it in config.openai_finetuning_dir.

    Return the path to the JSONL file.
    """
    os.makedirs(config.openai_finetuning_dir, exist_ok=True)
    jsonl = ""
    for row in dataset["conversation"]:
        jsonl += json.dumps({"messages": row}) + "\n"
    path = config.openai_finetuning_dir / file_name(len(dataset["conversation"]))
    with open(path, "w") as f:
        f.write(jsonl)
    return path


def upload_jsonl(path: Path):
    """
    Upload a JSONL file to OpenAI unless the filename already exists.
    """
    logger.info(f"Uploading {path} to OpenAI...")
    existing_files = client.files.list().data
    if any(f.filename == path.name for f in existing_files):
        logger.info(f"File {path.name} already exists in OpenAI, skipping")
        return
    client.files.create(file=open(path, "rb"), purpose="fine-tune")
    logger.info(f"Uploaded {path} to OpenAI")


def job_exists(
    existing_jobs: list[Any],
    model: str,
    file_id: str,
    lr_multiplier: float | int,
    batch_size: int,
    n_epochs: int,
) -> tuple[bool, str | None, str | None]:
    """
    Check if a job with these parameters already exists.

    Return a tuple of (exists, status, name of finetuned model).
    """
    matching_jobs = []
    for job in existing_jobs:
        if all(
            (
                job.model == model,
                job.training_file == file_id,
                job.hyperparameters.learning_rate_multiplier == lr_multiplier,
                job.hyperparameters.batch_size == batch_size,
                job.hyperparameters.n_epochs == n_epochs,
            )
        ):
            matching_jobs.append(job)

    if not matching_jobs:
        return False, None, None

    # In case there are multiple jobs, we return only the best one, based on its status:
    # "succeeded" > "running" = other > "pending" > "failed"
    def job_score(job: Any) -> int:
        match job.status:
            case "succeeded":
                return 3
            case "running":
                return 2
            case "pending":
                return 1
            case "failed":
                return 0
            case _:
                # Other statuses are considered to probably be akin to running
                return 2

    best_job = max(matching_jobs, key=job_score)
    return True, best_job.status, best_job.fine_tuned_model


def find_file_id(
    existing_files: list[Any], sample_size: int, error_if_not_found: bool = False
) -> str | None:
    for f in existing_files:
        if f.filename == file_name(sample_size):
            return f.id
    if error_if_not_found:
        raise RuntimeError(f"File {file_name(sample_size)} not found in OpenAI storage")
    return None


def generate_files(upload: bool = False):
    if DATASET_NAME == "lmsys-chat-1m":
        dataset = load_lmsys_chat_1m(
            gpt4_filter=True,
            redacted_filter=True,
            flagged_filter=True,
            first_turn_only=True,
            use_cache=True,
            datasets_dir=config.datasets_dir,
        )
    else:
        raise NotImplementedError(f"Dataset {DATASET_NAME} not implemented")
    indices = np.random.permutation(len(dataset))
    dataset = dataset.select(indices)
    for sample_size in SAMPLE_SIZES:
        subset = dataset.select(range(sample_size))
        jsonl_path = generate_jsonl(subset)
        if upload:
            upload_jsonl(jsonl_path)


def finetune(confirm: bool = False):
    existing_finetuning_jobs = client.fine_tuning.jobs.list(limit=10_000).data
    existing_files = client.files.list().data
    for sample_size in SAMPLE_SIZES:
        file_id = find_file_id(existing_files, sample_size, error_if_not_found=True)
        assert file_id is not None
        for model in MODELS:
            for lr_multiplier in LR_MULTIPLIERS:
                suffix = f"lr={lr_multiplier:g}_samples={sample_size}_epochs={EPOCHS}_batch_size={BATCH_SIZE}"
                logger.info(f"Creating finetuning job with {model}, {suffix=}...")
                exists, status, model_name = job_exists(
                    existing_finetuning_jobs, model, file_id, lr_multiplier, BATCH_SIZE, EPOCHS
                )
                if exists and status != "failed":
                    logger.info(f"Job already exists, {status=}, {model_name=}. Skipping")
                    continue
                if status == "failed":
                    logger.info(f"Job failed, {model_name=}. Retrying")
                hyperparameters = {
                    "learning_rate_multiplier": lr_multiplier,
                    "batch_size": BATCH_SIZE,
                    "n_epochs": EPOCHS,
                }
                if confirm and input("Are you sure you want to send this job? (y/N)") != "y":
                    logger.info("Skipping")
                    continue
                client.fine_tuning.jobs.create(
                    model=model,
                    training_file=file_id,
                    seed=config.seed,
                    suffix=suffix,
                    method={  # pyright: ignore[reportArgumentType]
                        "type": "supervised",
                        "supervised": {
                            "hyperparameters": hyperparameters,
                        },
                    },
                )
                logger.info("Finetuning job created")


def list_model_names():
    """
    List the names of the finetuned models in format suitable for copy-pasting into main.py.
    """
    finetuning_jobs = client.fine_tuning.jobs.list(limit=10_000).data
    existing_files = client.files.list().data
    endpoints = []
    for sample_size in SAMPLE_SIZES:
        file_id = find_file_id(existing_files, sample_size, error_if_not_found=True)
        assert file_id is not None
        for model in MODELS:
            for lr_multiplier in LR_MULTIPLIERS:
                suffix = f"lr={lr_multiplier:g}_samples={sample_size}_epochs={EPOCHS}_batch_size={BATCH_SIZE}"
                exists, status, model_name = job_exists(
                    finetuning_jobs, model, file_id, lr_multiplier, BATCH_SIZE, EPOCHS
                )
                if not exists:
                    logger.warning(f"Job with {model}, {suffix=} does not exist")
                elif not model_name:
                    logger.warning(
                        f"Job with {model}, {suffix=} does not yet have a name. {status=}"
                    )
                else:
                    if "gpt-4.1-2025-04-14" in model_name:
                        cost = (3, 12)
                    elif "gpt-4.1-mini-2025-04-14" in model_name:
                        cost = (0.8, 3.2)
                    elif "gpt-4.1-nano-2025-04-14" in model_name:
                        cost = (0.2, 0.8)
                    endpoints.append(f'    Endpoint("openai", "{model_name}", cost={cost}),')
    endpoints.sort()  # will sort by model name first
    print("\n".join(endpoints))


if __name__ == "__main__":
    # generate_files(upload=True)
    # finetune(confirm=False)
    list_model_names()
