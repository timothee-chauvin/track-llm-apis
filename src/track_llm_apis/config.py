import logging
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("track-llm-apis")
ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = Path(__file__).parent


class AnalysisConfig(BaseModel):
    cusum_warmup_period: int = 100
    cusum_threshold_probability: float = 1e-5
    ema_factor: float = 0.9


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


class SamplingConfig(BaseModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device_config: DeviceConfig = Field(default_factory=DeviceConfig)


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
        env_prefix="TRACK_LLM_APIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Paths
    root_dir: Path = Field(default_factory=lambda: ROOT_DIR)
    db_path: Path = Field(default_factory=lambda: ROOT_DIR / "db" / "llm_logprobs.db")
    plots_dir: Path = Field(default_factory=lambda: ROOT_DIR / "plots")
    data_dir: Path = Field(default_factory=lambda: ROOT_DIR / "data")
    openai_finetuning_dir: Path = Field(
        default_factory=lambda: ROOT_DIR / "data" / "openai_finetuning"
    )
    sampling_data_dir: Path = Field(default_factory=lambda: ROOT_DIR / "data" / "sampling")
    baselines_dir: Path = Field(default_factory=lambda: ROOT_DIR / "data" / "baselines")
    datasets_dir: Path = Field(default_factory=lambda: ROOT_DIR / "data" / "datasets")

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
    def logger(self) -> logging.Logger:
        return logger

    @computed_field
    @property
    def chat_templates(self) -> dict[str, Any]:
        with open(SRC_DIR / "chat_templates.toml", "rb") as f:
            return tomllib.load(f)

    @computed_field
    @property
    def date(self) -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @computed_field
    @property
    def last_commit_hash(self) -> str:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


config = Config()
