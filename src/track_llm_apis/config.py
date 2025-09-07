import json
import logging
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import torch
from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from track_llm_apis import get_assets_dir
from track_llm_apis.util import dataset_info, load_lmsys_chat_1m

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("track-llm-apis")


class CusumAnalysisConfig(BaseModel):
    warmup_period: int = 100
    threshold_probability: float = 1e-5
    ema_factor: float = 0.9


class AnalysisConfig(BaseModel):
    cusum: CusumAnalysisConfig = Field(default_factory=CusumAnalysisConfig)
    device: str | None = "cuda" if torch.cuda.is_available() else None


class DeviceConfig(BaseModel):
    device: str | None = None
    vllm_device: str | None = None
    original_model_device: str | None = None
    variants_device: str | None = None

    @model_validator(mode="after")
    def validate_device_config(self) -> Self:
        if (
            self.device is None
            and self.vllm_device is None
            and self.original_model_device is None
            and self.variants_device is None
        ):
            self.device = "cuda"
            self.vllm_device = self.device
            self.original_model_device = self.device
            self.variants_device = self.device
        elif self.device is not None:
            if (
                self.vllm_device is not None
                or self.original_model_device is not None
                or self.variants_device is not None
            ):
                raise ValueError(
                    "If 'device' is provided, 'vllm_device', 'original_model_device', and 'variants_device' must be None"
                )
            self.vllm_device = self.device
            self.original_model_device = self.device
            self.variants_device = self.device
        elif (
            self.vllm_device is None
            or self.original_model_device is None
            or self.variants_device is None
        ):
            raise ValueError(
                "Either provide 'device' alone, or all of 'vllm_device', 'original_model_device', and 'variants_device'"
            )

        return self


class Gao2025Config(BaseModel):
    n_wikipedia_prompts: int = 25
    n_wikipedia_samples_per_prompt: int = 10
    wikipedia_seed: int = 0
    max_tokens: int = 50
    temperature: float = 1.0


class MMLUConfig(BaseModel):
    subset_name: str = "abstract_algebra"
    max_tokens: int = 5
    temperature: float = 0.1
    # at the time of analysis, how many benchmark runs to use per p-value test
    n_samples_per_prompt: int = 10

    @property
    def answers(self) -> dict[str, int]:
        cache_path = config.data_dir / "mmlu_answers.json"
        try:
            with open(cache_path) as f:
                return json.load(f)
        except FileNotFoundError:
            # Download the dataset and create the cache file
            answers = {}
            mmlu = load_dataset("cais/mmlu", self.subset_name, split="test")
            for row in mmlu:
                answers[row["question"]] = row["answer"]
            with open(cache_path, "w") as f:
                json.dump(answers, f, indent=2)
            return answers


class LogprobConfig(BaseModel):
    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,  # for Dataset
    )
    batch_size: int = 64
    topk: int = 20
    temperature: float = 0.0
    # at the time of analysis, how many samples to use per p-value test
    n_samples_per_prompt: int = 10
    _other_prompts_dataset: Dataset | None = None

    @property
    def other_prompts_dataset(self) -> Dataset:
        if self._other_prompts_dataset is None:
            self._other_prompts_dataset = load_lmsys_chat_1m(
                use_cache=True, datasets_dir=config.datasets_dir
            )
        return self._other_prompts_dataset

    @property
    def other_prompts(self) -> list[str]:
        return [item["conversation"][0]["content"] for item in self.other_prompts_dataset]

    @computed_field
    @property
    def other_prompts_dataset_info(self) -> dict[str, str | int]:
        return dataset_info(self.other_prompts_dataset)


class SamplingConfig(BaseModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device_config: DeviceConfig = Field(default_factory=DeviceConfig)
    # How many times to sample the original model vs variants
    original_model_n_samples: int = 1_000
    variants_n_samples: int = 100
    vllm_enable_sleep_mode: bool = False

    gao2025: Gao2025Config = Field(default_factory=Gao2025Config)
    mmlu: MMLUConfig = Field(default_factory=MMLUConfig)
    logprob: LogprobConfig = Field(default_factory=LogprobConfig)


class APIConfig(BaseModel):
    top_logprobs: dict[str, int] = Field(
        default_factory=lambda: {
            "openai": 20,
            "grok": 8,
            "openrouter": 20,
        }
    )
    top_logprobs_openrouter: dict[str, int] = Field(
        default_factory=lambda: {
            # Default is 20, but these providers have a lower limit.
            "fireworks": 5,
            "azure": 5,
            "xai": 8,
        }
    )
    api_seed: int = Field(default=1, description="Grok API refuses a seed of 0, must be positive")
    max_retries: int = 15
    extended_endpoints_max_cost: float = Field(
        default=30.0, description="Sum of input and output costs per Mtok"
    )

    @field_validator("api_seed")
    @classmethod
    def validate_api_seed(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("api_seed must be positive (Grok API requirement)")
        return v


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        arbitrary_types_allowed=False,
        validate_assignment=True,
        env_prefix="TRACKLLM_",
        # e.g. specify the model name: TRACKLLM_SAMPLING__MODEL_NAME=...
        env_nested_delimiter="__",
        env_file=".env.config",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Paths
    data_dir: Path = Field(default_factory=lambda: Path("/data"))

    # Prompts to send to the smaller list of endpoints
    prompts: list[str] = Field(
        default_factory=lambda: [
            "x ",
            "x " * 5,
            "x " * 20,
            "Let's generate random words! Only output the words, no other text. Continue the list: Underpay\nPolicy\nRisotto\nIdealist",
            "Let's generate random words! Only output the words, no other text. Continue the list: Sinuous\nCornbread\nStipulate\nOverreact",
            "reply in one token. 1+1=",
            # Random prompts
            # 2 characters
            "]\n",
            "HB",
            "e|",
            # 4 characters
            "xég",
            "\x04B\x02z",
            "\x1e·T",
            # 8 characters
            "\x06P\x1dz\x13ZTq",
            "ZZ\x17˚p|[",
            "\x14\x1ap88V?_",
        ]
    )
    # Prompts to send to both the smaller and the extended lists of endpoints
    prompts_extended: list[str] = Field(default_factory=lambda: ["x"])

    max_completion_tokens: int = 1
    seed: int = 0

    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @property
    def assets_dir(self) -> Path:
        return get_assets_dir()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db" / "llm_logprobs.db"

    @property
    def plots_dir(self) -> Path:
        return self.data_dir / "plots"

    @property
    def openai_finetuning_dir(self) -> Path:
        return self.data_dir / "openai_finetuning"

    @property
    def sampling_data_dir(self) -> Path:
        return self.data_dir / "sampling"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def logger(self) -> logging.Logger:
        return logger

    @property
    def chat_templates(self) -> dict[str, Any]:
        with open(get_assets_dir() / "chat_templates.toml", "rb") as f:
            return tomllib.load(f)

    @computed_field
    @property
    def date(self) -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


config = Config()
