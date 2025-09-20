import gzip
import json
import os
import random
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import orjson
import rapidgzip
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score, roc_curve
from vllm import RequestOutput

from track_llm_apis.config import config
from track_llm_apis.util import slugify

logger = config.logger

# Type aliases for clarity
Condition = str
Variant = str
ROCCurve = tuple[list[float], list[float]]


class DataSource(Enum):
    US = 0
    MMLU = 1
    GAO2025 = 2


class References:
    def __init__(self):
        # Dictionaries mapping element => index of the element in the dictionary
        # which preserves insertion order
        self.variants: dict[str, int] = {}
        # (prompt, input_tokens)
        self.prompts: dict[tuple[str, int], int] = {}
        # (text, output_tokens)
        self.texts: dict[tuple[str, int], int] = {}
        self.logprobs: dict[str, int] = {}
        # Cache for the ordered keys of the dictionaries, for lookup by index
        self._cache = {}

    def to_json(self) -> dict[str, Any]:
        return {
            "variants": list(self.variants.keys()),
            "prompts": list(self.prompts.keys()),
            "texts": list(self.texts.keys()),
            "logprobs": list(self.logprobs.keys()),
        }

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> Self:
        instance = cls.__new__(cls)
        instance.variants = {variant: i for i, variant in enumerate(json_data["variants"])}
        instance.prompts = {prompt: i for i, prompt in enumerate(json_data["prompts"])}
        instance.texts = {text: i for i, text in enumerate(json_data["texts"])}
        instance.logprobs = {logprobs: i for i, logprobs in enumerate(json_data["logprobs"])}
        instance._cache = {}
        return instance

    def _get_keys(self, attr_name: str) -> list[Any]:
        if attr_name not in self._cache:
            self._cache[attr_name] = list(getattr(self, attr_name).keys())
        return self._cache[attr_name]

    def _invalidate_cache(self, attr_name: str):
        self._cache.pop(attr_name, None)

    def get_variant(self, variant_idx: int) -> str:
        return self._get_keys("variants")[variant_idx]

    def get_prompt(self, prompt_idx: int) -> tuple[str, int]:
        return self._get_keys("prompts")[prompt_idx]

    def get_text(self, text_idx: int) -> tuple[str, int]:
        return self._get_keys("texts")[text_idx]

    def get_logprobs(self, logprobs_idx: int) -> list[dict[int, float]]:
        raw = json.loads(self._get_keys("logprobs")[logprobs_idx])
        return [{int(k): v for k, v in lp.items()} for lp in raw]

    def add_variant(self, variant: str) -> int:
        if variant not in self.variants:
            self.variants[variant] = len(self.variants)
            self._invalidate_cache("variants")
        return self.variants[variant]

    def add_prompt(self, prompt: tuple[str, int]) -> int:
        if prompt not in self.prompts:
            self.prompts[prompt] = len(self.prompts)
            self._invalidate_cache("prompts")
        return self.prompts[prompt]

    def add_text(self, text: tuple[str, int]) -> int:
        if text not in self.texts:
            self.texts[text] = len(self.texts)
            self._invalidate_cache("texts")
        return self.texts[text]

    def add_logprobs(self, logprobs: list[dict[int, float]]) -> int:
        logprobs_str = json.dumps(logprobs)
        if logprobs_str not in self.logprobs:
            self.logprobs[logprobs_str] = len(self.logprobs)
            self._invalidate_cache("logprobs")
        return self.logprobs[logprobs_str]


class CompressedOutputRow:
    def __init__(
        self,
        references: References,
        source: int,
        variant_idx: int,
        prompt_idx: int,
        text_idx: int,
        logprobs_idx: int | None = None,
    ):
        """Initialization from indices"""
        self.references = references
        self.source = source
        self.variant_idx = variant_idx
        self.prompt_idx = prompt_idx
        self.text_idx = text_idx
        self.logprobs_idx = logprobs_idx

    @classmethod
    def from_values(
        cls,
        references: References,
        source: int,
        variant: str,
        prompt: tuple[str, int],
        text: tuple[str, int],
        logprobs: list[dict[int, float]] | None = None,
    ):
        """Initialization from values, adding to the references if necessary."""
        instance = cls.__new__(cls)
        instance.references = references
        instance.source = source
        instance.variant_idx = references.add_variant(variant)
        instance.prompt_idx = references.add_prompt(prompt)
        instance.text_idx = references.add_text(text)
        instance.logprobs_idx = references.add_logprobs(logprobs) if logprobs is not None else None
        return instance

    def to_json(self) -> tuple[int, int, int, int, int | None]:
        return (self.source, self.variant_idx, self.prompt_idx, self.text_idx, self.logprobs_idx)

    @classmethod
    def from_json(cls, references: References, json_data: Sequence[int | None]) -> Self:
        """This function assumes that the references are already up-to-date."""
        return cls(
            references=references,
            source=json_data[0],
            variant_idx=json_data[1],
            prompt_idx=json_data[2],
            text_idx=json_data[3],
            logprobs_idx=json_data[4],
        )

    @property
    def variant(self) -> str:
        return self.references.get_variant(self.variant_idx)

    @property
    def prompt(self) -> tuple[str, int]:
        return self.references.get_prompt(self.prompt_idx)

    @property
    def text(self) -> tuple[str, int]:
        return self.references.get_text(self.text_idx)

    @property
    def logprobs(self) -> list[dict[int, float]] | None:
        if self.logprobs_idx is None:
            return None
        return self.references.get_logprobs(self.logprobs_idx)


class CompressedOutput:
    def __init__(self, model_name: str, gpus: list[str] | None = None):
        self.model_name: str = model_name
        # GPUs used during sampling
        self.gpus: list[str] | None = gpus
        self.rows: list[CompressedOutputRow] = []
        self.references: References = References()

    def add_batch_from_request_output(
        self, request_output: RequestOutput, variant: str, source: DataSource
    ):
        prompt_length = (
            len(request_output.prompt_token_ids)
            if request_output.prompt_token_ids is not None
            else 0
        )
        prompt = (request_output.prompt or "", prompt_length)
        for output in request_output.outputs:
            if output.logprobs is None:
                logprobs_dicts = None
            else:
                logprobs_dicts = [
                    {int(k): v.logprob for k, v in logprobs.items()} for logprobs in output.logprobs
                ]
            self.rows.append(
                CompressedOutputRow.from_values(
                    references=self.references,
                    source=source,
                    variant=variant,
                    prompt=prompt,
                    text=(output.text, len(output.token_ids)),
                    logprobs=logprobs_dicts,
                )
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "model_name": self.model_name,
            "gpus": self.gpus,
            "rows": [row.to_json() for row in self.rows],
            "references": self.references.to_json(),
        }

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> Self:
        if not hasattr(json_data, "version"):
            # v0
            logger.info("Loading compressed output v0...")
            return cls._from_json_v0(json_data)
        if json_data["version"] == 1:
            logger.info("Loading compressed output v1...")
            return cls._from_json_v1(json_data)
        raise ValueError(f"Unknown version: {json_data['version']}")

    @classmethod
    def _from_json_v0(cls, json_data: dict[str, Any]) -> Self:
        json_references = json_data["references"]
        references = References.__new__(References)
        references.variants = {variant: i for i, variant in enumerate(json_references["variant"])}
        references.prompts = {
            tuple(prompt): i for i, prompt in enumerate(json_references["prompt"])
        }
        references.texts = {tuple(text): i for i, text in enumerate(json_references["text"])}
        references.logprobs = {
            json.dumps(logprobs): i for i, logprobs in enumerate(json_references["logprobs"])
        }
        references._cache = {}

        result = cls.__new__(cls)
        result.model_name = json_data["model_name"]
        result.gpus = json_data.get("gpus", None)
        result.rows = [CompressedOutputRow.from_json(references, row) for row in json_data["rows"]]
        result.references = references
        return result

    @classmethod
    def _from_json_v1(cls, json_data: dict[str, Any]) -> Self:
        references = References.from_json(json_data["references"])
        result = cls.__new__(cls)
        result.model_name = json_data["model_name"]
        result.gpus = json_data.get("gpus", None)
        result.rows = [CompressedOutputRow.from_json(references, row) for row in json_data["rows"]]
        result.references = references
        return result

    def dump_json(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        json_filename = f"{slugify(self.model_name, max_length=200, hash_length=0)}.json.gz"
        json_path = output_dir / json_filename
        json_dict = self.to_json()
        with gzip.open(json_path, "wt") as f:
            json.dump(json_dict, f, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json_dir(cls, json_dir: Path) -> Self:
        logger.info(f"Searching for a .json.gz file in {json_dir}...")
        # Find the only .json.gz file in the directory
        json_paths = list(json_dir.glob("*.json.gz"))
        if len(json_paths) != 1:
            raise ValueError(f"Expected 1 .json.gz file in {json_dir}, got {len(json_paths)}")
        json_path = json_paths[0]
        logger.info(f"Decompressing {json_path}...")
        with rapidgzip.open(json_path, parallelization=os.cpu_count()) as f:
            logger.info("Loading JSON data from contents of the file...")
            json_dict = orjson.loads(f.read())
        return cls.from_json(json_dict)

    def filter(self, datasource: DataSource, keep_references: bool = True) -> Self:
        """Return a new CompressedOutput with only the rows for the given datasource.
        If keep_references is True, the References object will be kept in the new CompressedOutput, and potentially shared by multiple CompressedOutput objects. Filtering will be faster.
        """
        if keep_references:
            new_compressed_output = self.__class__(model_name=self.model_name, gpus=self.gpus)
            new_compressed_output.references = self.references
            for row in self.rows:
                if row.source == datasource.value:
                    new_compressed_output.rows.append(row)
            return new_compressed_output
        else:
            new_compressed_output = self.__class__(model_name=self.model_name, gpus=self.gpus)
            for row in self.rows:
                if row.source == datasource.value:
                    new_compressed_output.rows.append(
                        CompressedOutputRow.from_values(
                            references=new_compressed_output.references,
                            source=row.source,
                            variant=row.variant,
                            prompt=row.prompt,
                            text=row.text,
                            logprobs=row.logprobs,
                        )
                    )
            return new_compressed_output

    def get_rows_by_variant(self) -> dict[str, list[CompressedOutputRow]]:
        rows_by_variant = defaultdict(list)
        for row in self.rows:
            rows_by_variant[row.variant].append(row)
        return dict(rows_by_variant)

    @staticmethod
    def get_rows_by_prompt(rows: list[CompressedOutputRow]) -> dict[str, list[CompressedOutputRow]]:
        rows_by_prompt = defaultdict(list)
        for row in rows:
            rows_by_prompt[row.prompt[0]].append(row)
        return dict(rows_by_prompt)


class CIResult(BaseModel):
    lower: float
    avg: float
    upper: float

    def __str__(self) -> str:
        return f"({self.lower}, {self.avg}, {self.upper})"


class TwoSampleTestResult(BaseModel):
    """Result of a single two-sample test, that returns a single statistic and a p-value."""

    pvalue: float | None = None
    statistic: float


class TwoSampleMultiTestResult(BaseModel):
    """Result of multiple two-sample tests on the same variant."""

    stats: list[float]
    pvalues: list[float] | None = None
    input_token_avg: float
    output_token_avg: float

    def power(self, alpha: float) -> list[float]:
        if self.pvalues is None:
            raise ValueError("pvalues is None, can't compute power")
        return sum(pvalue < alpha for pvalue in self.pvalues) / len(self.pvalues)

    def roc_curve(self, orig: "TwoSampleMultiTestResult") -> ROCCurve:
        y_true = [0] * len(orig.stats) + [1] * len(self.stats)
        y_pred = orig.stats + self.stats
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return list(fpr), list(tpr)

    @property
    def stat_avg(self) -> float:
        return sum(self.stats) / len(self.stats)

    @property
    def pvalue_avg(self) -> float:
        if self.pvalues is None:
            raise ValueError("pvalues is None, can't compute pvalue_avg")
        return sum(self.pvalues) / len(self.pvalues)

    @staticmethod
    def multivariant_roc(
        orig: "TwoSampleMultiTestResult", variants: list["TwoSampleMultiTestResult"]
    ) -> tuple[np.ndarray, np.ndarray]:
        y_true = [0] * len(orig.stats) + sum([[1] * len(variant.stats) for variant in variants], [])
        y_pred = orig.stats + [stat for variant in variants for stat in variant.stats]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return fpr, tpr

    @staticmethod
    def multivariant_roc_auc(
        orig: "TwoSampleMultiTestResult", variants: list["TwoSampleMultiTestResult"]
    ) -> float:
        y_true = [0] * len(orig.stats) + sum([[1] * len(variant.stats) for variant in variants], [])
        y_pred = orig.stats + [stat for variant in variants for stat in variant.stats]
        return roc_auc_score(y_true, y_pred)


# class TwoSampleMultiTestResultMultiROC(BaseModel):
#     """Result of multiple sets of multiple two-sample tests on the same variant, allowing to compute multiple ROC curves."""

#     results: list["TwoSampleMultiTestResultROC"]

#     @staticmethod
#     def _ci(values: list[float | None], results_alpha: float) -> CIResult:
#         if any(value is None for value in values):
#             return CIResult(
#                 lower=None,
#                 avg=None,
#                 upper=None,
#             )
#         values = sorted(values)
#         return CIResult(
#             lower=values[int(results_alpha * len(values))],
#             avg=sum(values) / len(values),
#             upper=values[int((1 - results_alpha) * len(values))],
#         )

#     def roc_auc_ci(
#         self, origs: "TwoSampleMultiTestResultMultiROC", results_alpha: float
#     ) -> CIResult:
#         roc_aucs = [result.roc_auc(orig) for orig, result in zip(origs.results, self.results)]
#         return self._ci(roc_aucs, results_alpha)

#     def roc_curves(self, origs: "TwoSampleMultiTestResultMultiROC") -> list[ROCCurve]:
#         return [result.roc_curve(orig) for orig, result in zip(origs.results, self.results)]

#     def stat_ci(self, results_alpha: float) -> CIResult:
#         return self._ci([result.stat_avg for result in self.results], results_alpha)

#     def pvalue_ci(self, results_alpha: float) -> CIResult:
#         return self._ci([result.pvalue_avg for result in self.results], results_alpha)

#     def power_ci(self, detector_alpha: float, results_alpha: float) -> CIResult:
#         return self._ci([result.power(detector_alpha) for result in self.results], results_alpha)

#     @staticmethod
#     def multivariant_rocs(
#         origs: "TwoSampleMultiTestResultMultiROC",
#         all_variants: list["TwoSampleMultiTestResultMultiROC"],
#     ) -> list[tuple[np.ndarray, np.ndarray]]:
#         return [
#             TwoSampleMultiTestResultROC.multivariant_roc(orig, variants)
#             for orig, variants in zip(
#                 origs.results, [variants.results for variants in all_variants]
#             )
#         ]

#     @staticmethod
#     def multivariant_roc_auc_ci(
#         origs: "TwoSampleMultiTestResultMultiROC",
#         all_variants: list["TwoSampleMultiTestResultMultiROC"],
#         results_alpha: float,
#     ) -> CIResult:
#         return TwoSampleMultiTestResultMultiROC._ci(
#             [
#                 TwoSampleMultiTestResultROC.multivariant_roc_auc(orig, variants)
#                 for orig, variants in zip(
#                     origs.results, [variants.results for variants in all_variants]
#                 )
#             ],
#             results_alpha,
#         )

#     @staticmethod
#     def roc_auc_avg_ci(
#         origs: "TwoSampleMultiTestResultMultiROC",
#         all_variants: list["TwoSampleMultiTestResultMultiROC"],
#         results_alpha: float,
#         return_values: bool = False,
#     ) -> CIResult:
#         """Return the average ROC AUC across all variants as a CI.

#         Suppose there are V variants, each containing S sets of two-sample tests (i.e. 1 TwoSampleMultiTestResultMultiRoc object, S TwoSampleMultiTestResultROC objects).
#         We compute S average ROC AUCs: for s in [1, ..., S], compute the average ROC AUC across V variants at set index s.
#         We then compute a CI on the list of S average ROC AUCs.

#         If return_values is True, return a tuple: (the CI, the list of S average ROC AUCs).
#         """
#         averages = []
#         S = len(all_variants[0].results)
#         for s in range(S):
#             averages.append(
#                 sum([variants.results[s].roc_auc(origs.results[s]) for variants in all_variants])
#                 / len(all_variants)
#             )
#         if return_values:
#             return TwoSampleMultiTestResultMultiROC._ci(averages, results_alpha), averages
#         else:
#             return TwoSampleMultiTestResultMultiROC._ci(averages, results_alpha)


class AnalysisResult(BaseModel):
    experiment: Literal["baseline", "ablation_prompt"]
    model_name: str
    n_tests: int
    pvalue_b: int

    original: dict[Condition, TwoSampleMultiTestResult] = {}
    variants: dict[Variant, dict[Condition, TwoSampleMultiTestResult]] = {}

    @property
    def input_token_avg(self) -> dict[Condition, float]:
        result = {}
        for condition in self.original.keys():
            result[condition] = sum(
                self.variants[variant][condition].input_token_avg for variant in self.variants
            ) / len(self.variants)
        return result

    @property
    def output_token_avg(self) -> dict[Condition, float]:
        result = {}
        for condition in self.original.keys():
            result[condition] = sum(
                self.variants[variant][condition].output_token_avg for variant in self.variants
            ) / len(self.variants)
        return result

    def compute_roc_curves(self, variant: Variant) -> dict[Condition, list[ROCCurve]]:
        raise NotImplementedError()

    def compute_roc_auc_ci(self, variant: Variant) -> dict[Condition, CIResult]:
        raise NotImplementedError()

    def avg_auc_across_variants(
        self, sampling: bool = False, centered: bool = False
    ) -> dict[Condition, float]:
        """If centered is True, subtract the average to the AUC of each condition.
        If sampling is True, sample with replacement from each list of statistics.
        """
        result = {}
        for condition in self.original.keys():
            result[condition] = sum(
                self.auc(variant=variant, condition=condition, sampling=sampling)
                for variant in self.variants
            ) / len(self.variants)

        if centered:
            avg_auc = sum(result.values()) / len(result)
            result = {condition: result[condition] - avg_auc for condition in result}

        return result

    def auc(self, variant: Variant, condition: Condition, sampling: bool) -> float:
        orig_stats = self.original[condition].stats
        variant_stats = self.variants[variant][condition].stats
        if sampling:
            orig_stats = random.choices(orig_stats, k=len(orig_stats))
            variant_stats = random.choices(variant_stats, k=len(variant_stats))
        y_true = [0] * len(orig_stats) + [1] * len(variant_stats)
        y_pred = orig_stats + variant_stats
        return roc_auc_score(y_true, y_pred)
