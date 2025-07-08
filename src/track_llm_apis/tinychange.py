import asyncio
import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TinyChangeConfig:
    enable_random_noise: bool = True
    random_noise_scale: list[float] = [2 ** (-n) for n in range(0, 16)]


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
        self.tasks = [self.random_noise(scale) for scale in self.config.random_noise_scale]
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
