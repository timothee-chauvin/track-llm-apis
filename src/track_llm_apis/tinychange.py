import asyncio
import copy
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from dotenv import load_dotenv
from torch.nn.utils import prune
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


class TinyChangeConfig:
    seed: int | None = 0
    enable_random_noise: bool = True
    random_noise_scale: list[float] = [2 ** (-n) for n in range(0, 16)]
    enable_weight_pruning: bool = True
    weight_pruning_magnitude_scale: list[float] = [float(2 ** (-n)) for n in range(0, 11)]
    weight_pruning_random_scale: list[float] = [float(2 ** (-n)) for n in range(0, 11)]


@dataclass
class TinyChangeModel:
    description: str
    model_hash: str
    model: Any


class TinyChange:
    # TODO in order or random order
    # TODO device handling, parallel processing on multiple GPUs
    # TODO reproducibility
    def __init__(self, model, tokenizer, config: TinyChangeConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        self.tasks = []
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
        self._task_index = 0

    @property
    def n_variants(self) -> int:
        """The number of modified models to generate."""
        return len(self.tasks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        """Generate and return the next modified model."""
        if self._task_index >= len(self.tasks):
            raise StopAsyncIteration
        task = self.tasks[self._task_index]
        self._task_index += 1
        return await task

    async def get_unchanged(self) -> TinyChangeModel:
        """Return the unchanged model as a TinyChangeModel object."""
        return TinyChangeModel(
            description="unchanged",
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
            description=f"random_noise_{scale:.2e}",
            model_hash=get_model_hash(model),
            model=model,
        )

    async def weight_pruning(
        self, scale: float, method: Literal["magnitude", "random"]
    ) -> TinyChangeModel:
        """Prune the model by removing parameters across the weights (not biases) of all MLP layers."""
        model = copy.deepcopy(self.model)
        if method == "magnitude":
            self.prune_llm(model, prune.L1Unstructured, scale)
        elif method == "random":
            self.prune_llm(model, prune.RandomUnstructured, scale)
        return TinyChangeModel(
            description=f"weight_pruning_{method}_{scale:.2e}",
            model_hash=get_model_hash(model),
            model=model,
        )

    @staticmethod
    def prune_llm(
        model: torch.nn.Module,
        pruning_method: type,
        amount: int | float,
    ) -> None:
        """
        Globally prune a given LLM in-place by removing parameters across the weights (not biases) of all MLP layers.

        Args:
            model: The LLM model to prune
            pruning_method: Pruning method (e.g., L1Unstructured, RandomUnstructured)
            amount: Fraction (0.0-1.0) or absolute number of parameters to prune
        """
        parameters_to_prune = []

        for name, module in model.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                if param_name == "weight" and "mlp" in name:
                    parameters_to_prune.append((module, param_name))

        prune.global_unstructured(parameters_to_prune, pruning_method=pruning_method, amount=amount)

        # Remove the name+'_orig' parameter and name+'_mask' buffer from all modified parameters
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)


def get_model_hash(model):
    """
    Compute a hash of the model's parameters.

    Args:
        model: PyTorch model

    Returns:
        str: Hexadecimal hash string representing the model state
    """
    hasher = hashlib.sha256()

    for _, param in sorted(model.named_parameters()):
        # Convert to float32 before converting to bytes to ensure consistent hashing
        param_data = param.detach().cpu().to(torch.float32).numpy().tobytes()
        hasher.update(param_data)

    return hasher.hexdigest()


async def main():
    # Testing
    DEVICE = "cuda"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tiny_change = TinyChange(model, tokenizer, TinyChangeConfig())

    print(await tiny_change.get_unchanged())

    # Synchronous iteration for testing
    async_iter = tiny_change.__aiter__()
    try:
        while True:
            variant = await async_iter.__anext__()
            print(f"Generated variant: {variant.description} ({variant.model_hash})")
    except StopAsyncIteration:
        print("All variants processed")


if __name__ == "__main__":
    asyncio.run(main())
