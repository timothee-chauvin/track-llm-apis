import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

from track_llm_apis.config import Config
from track_llm_apis.util import load_lmsys_chat_1m

random.seed(Config.seed)
np.random.seed(Config.seed)

logger = Config.logger

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


TESTING = True
if TESTING:
    MODELS = ["gpt-4.1-nano-2025-04-14"]
    LR_MULTIPLIERS = [1e-4, 1e-2, 1]
    SAMPLE_SIZES = [10, 20, 40, 80]


def file_name(sample_size: int) -> str:
    return f"{DATASET_NAME}_{sample_size}_seed={Config.seed}.jsonl"


def generate_jsonl(dataset: Dataset) -> Path:
    """
    Generate a JSONL file for OpenAI from `dataset`, save it in Config.openai_finetuning_dir.

    Return the path to the JSONL file.
    """
    os.makedirs(Config.openai_finetuning_dir, exist_ok=True)
    jsonl = ""
    for row in dataset["conversation"]:
        jsonl += json.dumps({"messages": row}) + "\n"
    path = Config.openai_finetuning_dir / file_name(len(dataset["conversation"]))
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
    lr_multiplier: float,
    batch_size: int,
    n_epochs: int,
) -> str | None:
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
            return job.fine_tuned_model
    return None


def main(gen_files: bool = False, upload_files: bool = False, finetune: bool = False):
    if upload_files:
        assert gen_files, "Must generate files before uploading"
    if gen_files:
        if DATASET_NAME == "lmsys-chat-1m":
            dataset = load_lmsys_chat_1m(
                gpt4_filter=True, redacted_filter=True, flagged_filter=True, first_turn_only=True
            )
        else:
            raise NotImplementedError(f"Dataset {DATASET_NAME} not implemented")
        indices = np.random.permutation(len(dataset))
        dataset = dataset.select(indices)
        for sample_size in SAMPLE_SIZES:
            subset = dataset[:sample_size]
            jsonl_path = generate_jsonl(subset)
            if upload_files:
                upload_jsonl(jsonl_path)

    if finetune:
        existing_finetuning_jobs = client.fine_tuning.jobs.list().data
        for sample_size in SAMPLE_SIZES:
            existing_files = client.files.list().data
            for f in existing_files:
                if f.filename == file_name(sample_size):
                    file_id = f.id
                    break
            for model in MODELS:
                for lr_multiplier in LR_MULTIPLIERS:
                    suffix = f"lr={lr_multiplier:g}_samples={sample_size}_epochs={EPOCHS}_batch_size={BATCH_SIZE}"
                    logger.info(f"Creating finetuning job with {model}, {suffix=}...")
                    # Find if a job with these parameters already exists
                    if job_name := job_exists(
                        existing_finetuning_jobs, model, file_id, lr_multiplier, BATCH_SIZE, EPOCHS
                    ):
                        logger.info(f"Job already exists with name {job_name}. Skipping")
                        continue
                    hyperparameters = {
                        "learning_rate_multiplier": lr_multiplier,
                        "batch_size": BATCH_SIZE,
                        "n_epochs": EPOCHS,
                    }
                    if input("Are you sure you want to send this job? (y/N)") != "y":
                        logger.info("Skipping")
                        continue
                    client.fine_tuning.jobs.create(
                        model=model,
                        training_file=file_id,
                        seed=Config.seed,
                        suffix=suffix,
                        method={
                            "type": "supervised",
                            "supervised": {
                                "hyperparameters": hyperparameters,
                            },
                        },
                    )
                    logger.info("Finetuning job created")


if __name__ == "__main__":
    # main(gen_files=True, upload_files=True)
    main(finetune=True)
