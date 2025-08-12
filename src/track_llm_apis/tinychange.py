import asyncio
import gc
import json
import logging
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lightning as pl
import modelopt.torch.quantization as mtq
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# TODO remove these dependencies
from track_llm_apis.util import (
    copy_model_to,
    get_dataset_hash,
    get_model_hash,
    load_lmsys_chat_1m,
    slugify,
    trim_to_length,
)

load_dotenv()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("tinychange")

ROOT_DIR = Path(__file__).parent.parent.parent


class TinyChangeConfig(BaseSettings):
    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,  # for Dataset
        validate_assignment=True,
        env_prefix="TINYCHANGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    datasets_dir: Path = Field(default_factory=lambda: ROOT_DIR / "data" / "datasets")

    variants_device: str | None = None
    enable_random_noise: bool = True
    enable_weight_pruning: bool = True
    enable_finetuning: bool = True
    enable_lora_finetuning: bool = True
    enable_quantization: bool = True
    seed: int | None = 0

    random_noise_scale: list[float | int] = Field(
        default_factory=lambda: [float(2 ** (-n)) for n in range(0, 16)]
    )
    weight_pruning_magnitude_scale: list[float | int] = Field(
        default_factory=lambda: [float(2 ** (-n)) for n in range(0, 11)]
    )
    weight_pruning_random_scale: list[float | int] = Field(
        default_factory=lambda: [float(2 ** (-n)) for n in range(0, 11)]
    )

    finetuning_dataset: Dataset | None = None
    finetuning_lr_scale: list[float | int] = Field(default_factory=lambda: [1e-6])
    finetuning_samples: list[int] = Field(default_factory=lambda: [2**n for n in range(0, 10)])
    finetuning_epochs: int = 1
    finetuning_batch_size: int = 1
    finetuning_max_length: int = 1024

    lora_r: int = 1
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["o_proj"]  # , "k_proj", "v_proj", "o_proj"]
    )

    quantization_methods: list[str] = Field(default_factory=lambda: ["int8"])

    @model_validator(mode="after")
    def trim_finetuning_dataset(self):
        """Trim the finetuning dataset to the max samples"""
        if self.finetuning_dataset is not None:
            max_samples = max(self.finetuning_samples)
            # Use object.__setattr__ to bypass validation and avoid recursion
            object.__setattr__(
                self, "finetuning_dataset", self.finetuning_dataset.select(range(max_samples))
            )
        return self

    @computed_field
    @property
    def finetuning_dataset_info(self) -> dict[str, Any]:
        if self.finetuning_dataset is None:
            return {
                "length": 0,
                "hash": "",
                "first": None,
                "last": None,
            }
        else:
            return {
                "length": len(self.finetuning_dataset),
                "hash": get_dataset_hash(self.finetuning_dataset),
                "first": trim_to_length(
                    self.finetuning_dataset[0]["conversation"][0]["content"], 100
                ),
                "last": trim_to_length(
                    self.finetuning_dataset[-1]["conversation"][-1]["content"], 100
                ),
            }

    @computed_field
    @property
    def finetuning_dataset_hash(self) -> str:
        if self.finetuning_dataset is None:
            return ""
        return get_dataset_hash(self.finetuning_dataset)


class LightningModel(pl.LightningModule):
    def __init__(self, model, lr: float | int):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        assistant_tokens_mask = batch["assistant_tokens_mask"]
        labels[assistant_tokens_mask == 0] = -100
        logits = self.model(input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return optimizer


class FinetuningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: TinyChangeConfig,
        n_samples: int,
        batch_size: int,
        tokenizer,
    ):
        super().__init__()
        self.config = config
        self.dataset = config.finetuning_dataset
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.tokenizer = tokenizer
        self.ft_max_length = config.finetuning_max_length

    def prepare_data(self):
        """Validate the finetuning dataset and create a random order of the finetuning samples, preprocess it for pytorch, save to disk."""
        logger.info("Preprocessing finetuning dataset...")
        disk_path = self._get_disk_path()
        if disk_path.exists():
            logger.info(f"Finetuning dataset already exists at {disk_path}, nothing to do.")
            return
        logger.info(f"Finetuning dataset does not exist at {disk_path}, preprocessing...")
        init_ds = self.dataset
        # assert isinstance(init_ds, Dataset)
        if len(init_ds) < self.n_samples:
            raise ValueError(
                f"Not enough finetuning samples ({len(init_ds)}) to finetune with {self.n_samples} samples"
            )
        assert "conversation" in init_ds.column_names, (
            "Finetuning dataset must have a 'conversation' column"
        )
        indices = np.random.permutation(len(init_ds))
        ft_ds = init_ds.select(indices)
        assert all(s["conversation"][0]["role"] == "user" for s in ft_ds), (  # pyright: ignore[reportArgumentType,reportCallIssue]
            "Finetuning dataset must have a 'user' role in the first message of each conversation"
        )
        assert all(s["conversation"][1]["role"] == "assistant" for s in ft_ds), (  # pyright: ignore[reportArgumentType,reportCallIssue]
            "Finetuning dataset must have an 'assistant' role in the second message of each conversation"
        )

        # Apply chat template
        def _preprocess(x):
            chat_template_result = self.tokenizer.apply_chat_template(
                x["conversation"],
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                padding="longest",
                max_length=self.ft_max_length,
            )
            return {
                "input_ids": chat_template_result["input_ids"],
                "attention_mask": chat_template_result["attention_mask"],
                "labels": chat_template_result["input_ids"],
                "assistant_tokens_mask": chat_template_result["assistant_masks"],
            }

        ft_ds = ft_ds.map(_preprocess, batched=True, batch_size=None)
        # keep only the new columns
        ft_ds = ft_ds.remove_columns(
            [
                col
                for col in ft_ds.column_names
                if col not in ["input_ids", "attention_mask", "labels", "assistant_tokens_mask"]
            ]
        )
        ft_ds = ft_ds.with_format("torch")
        # Save to disk
        ft_ds.save_to_disk(str(disk_path))

    def setup(self, stage: str):
        disk_path = self._get_disk_path()
        self.tokenized_dataset = load_from_disk(str(disk_path))
        assert isinstance(self.tokenized_dataset, Dataset)
        self.tokenized_dataset = self.tokenized_dataset.select(range(self.n_samples))

    def train_dataloader(self):
        return DataLoader(self.tokenized_dataset, batch_size=self.batch_size, shuffle=False)  # pyright: ignore[reportArgumentType]

    def _get_disk_path(self):
        dataset_hash = get_dataset_hash(self.dataset)
        tokenizer_slug = slugify(self.tokenizer.name_or_path, hash_length=0)
        disk_path = self.config.datasets_dir / dataset_hash / tokenizer_slug
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        return disk_path


@dataclass
class TinyChangeModel:
    description: dict[str, Any]
    model_hash: str
    model: Any

    def name(self) -> str:
        return json.dumps(self.description, separators=(",", ":"))


class TinyChange:
    # TODO in order or random order
    # TODO device handling, parallel processing on multiple GPUs
    # TODO reproducibility
    def __init__(self, model, tokenizer, config: TinyChangeConfig):
        self.model = model
        self.model_hash = get_model_hash(model)
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = config
        # Default to using the same device as the model for the variants
        if self.config.variants_device is None:
            self.config.variants_device = model.device

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False
            torch.use_deterministic_algorithms(True, warn_only=False)
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            pl.seed_everything(self.config.seed)

        self.tasks = []
        # Start by always returning the unchanged model
        self.tasks.append(self.get_unchanged())
        if self.config.enable_quantization:
            self.tasks.extend(
                [self.quantize(method) for method in self.config.quantization_methods]
            )
        if self.config.enable_random_noise:
            self.tasks.extend(
                [self.random_noise(scale) for scale in self.config.random_noise_scale]
            )
        if self.config.enable_weight_pruning:
            self.tasks.extend(
                [
                    self.weight_pruning(scale, method="magnitude")
                    for scale in self.config.weight_pruning_magnitude_scale
                ]
            )
            self.tasks.extend(
                [
                    self.weight_pruning(scale, method="random")
                    for scale in self.config.weight_pruning_random_scale
                ]
            )

        def generate_finetuning_tasks(use_lora: bool):
            return [
                self.finetune(
                    lr,
                    n_samples,
                    self.config.finetuning_epochs,
                    self.config.finetuning_batch_size,
                    use_lora=use_lora,
                )
                for lr in self.config.finetuning_lr_scale
                for n_samples in self.config.finetuning_samples
            ]

        if self.config.enable_finetuning:
            self.tasks.extend(generate_finetuning_tasks(use_lora=False))

        if self.config.enable_lora_finetuning:
            self.tasks.extend(generate_finetuning_tasks(use_lora=True))

        self._task_index = 0

    @property
    def n_variants(self) -> int:
        """The number of modified models to generate."""
        return len(self.tasks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        """Generate and return the next modified model."""
        gc.collect()
        torch.cuda.empty_cache()
        if self._task_index >= len(self.tasks):
            raise StopAsyncIteration
        task = self.tasks[self._task_index]
        self._task_index += 1
        result = await task
        gc.collect()
        torch.cuda.empty_cache()
        assert get_model_hash(self.model) == self.model_hash, (
            f"Original model hash changed from {self.model_hash} to {get_model_hash(self.model)}!!"
        )
        return result

    async def get_unchanged(self) -> TinyChangeModel:
        """Return the unchanged model as a TinyChangeModel object."""
        return TinyChangeModel(
            description={"type": "unchanged"},
            model_hash=get_model_hash(self.model),
            model=self.model,
        )

    async def random_noise(self, scale: float) -> TinyChangeModel:
        """Add random noise following a normal distribution of mean 0 and standard deviation `scale` to each parameter in the model."""
        model = copy_model_to(self.model, self.config.variants_device)  # pyright: ignore[reportArgumentType]
        for _, param in model.named_parameters():
            if param.requires_grad:
                param.data += torch.normal(
                    mean=0.0,
                    std=scale,
                    size=param.data.shape,
                    device=param.data.device,
                    dtype=param.data.dtype,
                )
        return TinyChangeModel(
            description={"type": "random_noise", "scale": scale},
            model_hash=get_model_hash(model),
            model=model,
        )

    async def weight_pruning(
        self, scale: float | int, method: Literal["magnitude", "random"]
    ) -> TinyChangeModel:
        """Prune the model by removing parameters across the weights (not biases) of all MLP layers."""
        model = copy_model_to(self.model, self.config.variants_device)  # pyright: ignore[reportArgumentType]

        if method == "magnitude":
            self.prune_llm(model, prune.l1_unstructured, scale)
        elif method == "random":
            self.prune_llm(model, prune.random_unstructured, scale)

        result = TinyChangeModel(
            description={"type": "weight_pruning", "method": method, "scale": scale},
            model_hash=get_model_hash(model),
            model=model,
        )

        return result

    @staticmethod
    def prune_llm(
        model: torch.nn.Module,
        pruning_method: Callable[..., None],
        amount: int | float,
    ) -> None:
        """
        Globally prune a given LLM in-place by removing parameters across the weights (not biases) of all MLP layers.
        This is done layer by layer to reduce memory overhead.

        Args:
            model: The LLM model to prune
            pruning_method: Pruning method (e.g., L1Unstructured, RandomUnstructured)
            amount: Fraction (0.0-1.0) or absolute number of parameters to prune
        """
        for name, module in model.named_modules():
            if "mlp" in name:
                # Pre-compute to avoid the dictionary keys changing during iteration due to pruning
                param_names = [
                    param_name
                    for param_name, _ in module.named_parameters(recurse=False)
                    if param_name == "weight"
                ]
                for param_name in param_names:
                    pruning_method(module, param_name, amount=amount)
                    prune.remove(module, param_name)

    async def finetune(
        self, lr: float | int, n_samples: int, epochs: int, batch_size: int, use_lora: bool = False
    ) -> TinyChangeModel:
        """Finetune the model on a subset of the finetuning dataset."""
        model = copy_model_to(self.model, self.config.variants_device)  # pyright: ignore[reportArgumentType]
        if use_lora:
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
        pl_model = LightningModel(model, lr=lr)

        assert self.config.finetuning_dataset is not None

        datamodule = FinetuningDataModule(
            config=self.config,
            n_samples=n_samples,
            batch_size=batch_size,
            tokenizer=self.tokenizer,
        )
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="cuda",
            devices=[int(self.config.variants_device[5:])],  # pyright: ignore[reportOptionalSubscript]
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        trainer.fit(model=pl_model, datamodule=datamodule)

        trained_model = pl_model.model

        if use_lora:
            trained_model = trained_model.merge_and_unload()  # pyright: ignore[reportCallIssue]

        # Recreate model with original tensor storage format
        recreated_model = copy_model_to(
            trained_model, self.config.variants_device, dtype=trained_model.dtype
        )
        del trained_model

        return TinyChangeModel(
            description={
                "type": "finetune",
                "lr": lr,
                "n_samples": n_samples,
                "lora": use_lora,
            },
            model_hash=get_model_hash(recreated_model),
            model=recreated_model,
        )

    async def quantize(self, method: str) -> TinyChangeModel:
        model = copy_model_to(self.model, self.config.variants_device)  # pyright: ignore[reportArgumentType]
        configs = {
            "int8": mtq.INT8_DEFAULT_CFG,
        }
        if method not in configs:
            raise ValueError(f"Unsupported quantization method: {method}")
        config = configs[method]
        model = mtq.quantize(model, config, forward_loop=None)
        return TinyChangeModel(
            description={"type": "quantization", "method": method},
            model_hash=get_model_hash(model),
            model=model,
        )


async def main():
    # Testing
    DEVICE = "cuda"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = TinyChangeConfig()
    if config.enable_finetuning or config.enable_lora_finetuning:
        config.finetuning_dataset = load_lmsys_chat_1m(
            gpt4_filter=True, redacted_filter=True, flagged_filter=True, first_turn_only=True
        )
    tiny_change = TinyChange(model, tokenizer, config)

    logger.info(f"dtype: {model.dtype}")
    logger.info(f"Base model hash: {(await tiny_change.get_unchanged()).model_hash}")

    # Synchronous iteration for testing
    async_iter = tiny_change.__aiter__()
    try:
        while True:
            variant = await async_iter.__anext__()
            logger.info(f"Generated variant: ({variant.model_hash})")
            logger.info(json.dumps(variant.description, indent=2))
    except StopAsyncIteration:
        logger.info("All variants processed")


if __name__ == "__main__":
    asyncio.run(main())
