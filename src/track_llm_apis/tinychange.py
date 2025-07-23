import asyncio
import copy
import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Literal

import modelopt.torch.quantization as mtq
import numpy as np
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from torch.nn.utils import prune
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from track_llm_apis.util import load_lmsys_chat_1m

load_dotenv()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("tinychange")


class TinyChangeConfig:
    enable_random_noise: bool = True
    enable_weight_pruning: bool = True
    enable_finetuning: bool = True
    enable_lora_finetuning: bool = True
    enable_quantization: bool = True

    seed: int | None = 0
    random_noise_scale: list[float] = [2 ** (-n) for n in range(0, 16)]
    weight_pruning_magnitude_scale: list[float] = [float(2 ** (-n)) for n in range(0, 11)]
    weight_pruning_random_scale: list[float] = [float(2 ** (-n)) for n in range(0, 11)]
    finetuning_dataset: Dataset | None = None
    finetuning_lr_scale: list[float] = [10 ** (-n) for n in range(3, 9)]
    finetuning_samples: list[int] = [2**n for n in range(0, 11)]
    finetuning_epochs: int = 1
    finetuning_batch_size: int = 1
    finetuning_max_length: int = 1024
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    quantization_methods: list[str] = ["int8"]


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
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = config
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
        if self.config.enable_finetuning or self.config.enable_lora_finetuning:
            self.init_finetuning()

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
        torch.cuda.empty_cache()
        if self._task_index >= len(self.tasks):
            raise StopAsyncIteration
        task = self.tasks[self._task_index]
        self._task_index += 1
        return await task

    def init_finetuning(self):
        """Validate the finetuning dataset and create a random order of the finetuning samples, and preprocess it for pytorch"""
        init_ds = self.config.finetuning_dataset
        if len(init_ds) < max(self.config.finetuning_samples):
            raise ValueError(
                f"Not enough finetuning samples ({len(init_ds)}) to finetune with {max(self.config.finetuning_samples)} samples"
            )
        assert "conversation" in init_ds.column_names, (
            "Finetuning dataset must have a 'conversation' column"
        )
        assert all(s["conversation"][0]["role"] == "user" for s in init_ds), (
            "Finetuning dataset must have a 'user' role in the first message of each conversation"
        )
        assert all(s["conversation"][1]["role"] == "assistant" for s in init_ds), (
            "Finetuning dataset must have an 'assistant' role in the second message of each conversation"
        )
        indices = np.random.permutation(len(init_ds))
        ft_ds = init_ds.select(indices)
        # Apply chat template
        ft_ds = ft_ds.map(
            lambda x: {
                "text": self.tokenizer.apply_chat_template(
                    x["conversation"], tokenize=False, add_generation_prompt=False
                )
            },
            batched=False,
        )
        self.finetuning_ds = ft_ds

    async def get_unchanged(self) -> TinyChangeModel:
        """Return the unchanged model as a TinyChangeModel object."""
        return TinyChangeModel(
            description={"type": "unchanged"},
            model_hash=get_model_hash(self.model),
            model=self.model,
        )

    async def random_noise(self, scale: float) -> TinyChangeModel:
        """Add random noise following a normal distribution of mean 0 and standard deviation `scale` to each parameter in the model."""
        model = copy.deepcopy(self.model)
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
        self, scale: float, method: Literal["magnitude", "random"]
    ) -> TinyChangeModel:
        """Prune the model by removing parameters across the weights (not biases) of all MLP layers."""
        model = copy.deepcopy(self.model)

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
        pruning_method: type,
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
        self, lr: float, n_samples: int, epochs: int, batch_size: int, use_lora: bool = False
    ) -> TinyChangeModel:
        """Finetune the model on a subset of the finetuning dataset."""
        model = copy.deepcopy(self.model)

        subset = self.finetuning_ds.select(range(n_samples))
        training_args = SFTConfig(
            save_strategy="no",
            dataset_text_field="text",
            max_length=self.config.finetuning_max_length,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            lr_scheduler_type="constant",
            completion_only_loss=True,
        )

        sft_trainer_args = {
            "model": model,
            "args": training_args,
            "train_dataset": subset,
        }
        if use_lora:
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                task_type="CAUSAL_LM",
            )
            sft_trainer_args["peft_config"] = peft_config

        trainer = SFTTrainer(**sft_trainer_args)
        trainer.train()
        return TinyChangeModel(
            description={
                "type": "finetune",
                "lr": lr,
                "n_samples": n_samples,
                "lora": use_lora,
            },
            model_hash=get_model_hash(model),
            model=model,
        )

    async def quantize(self, method: str) -> TinyChangeModel:
        model = copy.deepcopy(self.model)
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


def get_model_hash(model):
    """
    Compute a hash of the model's parameters.

    Args:
        model: PyTorch model

    Returns:
        str: Hexadecimal hash string representing the model state
    """
    hasher = hashlib.sha256()

    # Parameters
    for _, param in sorted(model.named_parameters()):
        # Convert to float32 before converting to bytes to ensure consistent hashing
        param_data = param.detach().cpu().to(torch.float32).numpy().tobytes()
        hasher.update(param_data)

    # Buffers
    for _, buffer in sorted(model.named_buffers()):
        if buffer is not None:
            buffer_data = buffer.detach().cpu().to(torch.float32).numpy().tobytes()
            hasher.update(buffer_data)

    return hasher.hexdigest()


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
