import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import orjson
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from pydantic import BaseModel, field_validator
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from track_llm_apis.config import config
from track_llm_apis.sampling.analyze_gao2025 import CompletionSample, run_two_sample_test_torch
from track_llm_apis.sampling.analyze_logprobs import logprob_two_sample_test
from track_llm_apis.sampling.analyze_mmlu import mmlu_two_sample_test
from track_llm_apis.sampling.common import (
    AnalysisResult,
    CIResult,
    CompressedOutput,
    CompressedOutputRow,
    DataSource,
    TwoSampleMultiTestResult,
    Variant,
)
from track_llm_apis.tinychange import TinyChange
from track_llm_apis.util import ci, slugify, trim_to_length

logger = config.logger


def get_logprobs_transformers(model, tokenizer, prompt, model_device):
    """Get log probabilities for the first generated token using model.generate() from transformers."""
    full_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model_device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )
        first_token_logits = outputs.scores[0]  # Shape: [batch_size, vocab_size]
        first_token_logprobs = torch.log_softmax(first_token_logits, dim=-1)
        return first_token_logprobs[0]


def compute_kl_divergence(original_logprobs, variant_logprobs):
    """Compute KL divergence between two log probability distributions."""
    return F.kl_div(variant_logprobs, original_logprobs, log_target=True, reduction="mean")


def print_logprobs_summary(logprobs, tokenizer, model_name):
    """logger.info summary of top logprobs for a model."""
    top_10_logprobs, top_10_indices = torch.topk(logprobs, 10)

    logger.info(f"\nTop 10 tokens for {model_name}:")
    for i in range(10):
        token_id = top_10_indices[i].item()
        token = tokenizer.decode(token_id)
        logprob = top_10_logprobs[i].item()
        logger.info(f"Rank {i + 1}: '{token}' (ID: {token_id}) - Log prob: {logprob:g}")


def print_logprobs(model, tokenizer, prompt, model_device):
    full_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model_device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )

        first_token_logits = outputs.scores[0]  # Shape: [batch_size, vocab_size]
        first_token_logprobs = torch.log_softmax(first_token_logits, dim=-1)
        top_10_logprobs, top_10_indices = torch.topk(first_token_logprobs[0], 10)

        logger.info("Top 10 tokens and their log probabilities:")
        for i in range(10):
            token_id = top_10_indices[i].item()
            token = tokenizer.decode(token_id)
            logprob = top_10_logprobs[i].item()
            logger.info(f"Rank {i + 1}: '{token}' (ID: {token_id}) - Log prob: {logprob:g}")

        # Also logger.info the actually generated token for reference
        generated_token_id = outputs.sequences[0, -1].item()
        generated_token = tokenizer.decode(generated_token_id)
        logger.info(f"\nActually generated token: '{generated_token}' (ID: {generated_token_id})")


def plot_logprobs_over_time(
    all_logprobs,
    prompt: str,
    base_model_name: str,
    variant_description: dict[str, Any],
    batch_size: int,
):
    """Plot logprobs over time for all tokens that appear in the series."""
    prompt_slug = slugify(prompt, max_length=50, hash_length=8)
    model_name_slug = slugify(base_model_name, hash_length=0)
    prompt_dir = (
        config.plots_dir
        / "time_series_local"
        / prompt_slug
        / model_name_slug
        / f"batch_size={batch_size}"
    )
    os.makedirs(prompt_dir, exist_ok=True)
    description_str = "_".join(str(v) for v in variant_description.values())
    description_slug = slugify(description_str, max_length=100, hash_length=8)
    filename = f"{description_slug}.html"

    # Collect all unique tokens
    all_tokens = set()
    for logprob_dict in all_logprobs:
        all_tokens.update(logprob_dict.keys())

    fig = go.Figure()  # pyright: ignore[reportCallIssue]

    # For each token, create its time series
    for token_id in sorted(all_tokens):
        logprob_series = []

        for logprob_dict in all_logprobs:
            if token_id in logprob_dict:
                logprob_obj = logprob_dict[token_id]
                decoded_token = logprob_obj.decoded_token
                logprob_series.append(logprob_obj.logprob)
            else:
                logprob_series.append(None)

        fig.add_trace(
            go.Scatter(  # pyright: ignore[reportCallIssue]
                x=list(range(len(all_logprobs))),
                y=logprob_series,
                mode="lines+markers",
                name=decoded_token,
                connectgaps=False,
            )
        )

    prompt_preview = repr(trim_to_length(prompt, 50))
    fig.update_layout(
        title=f"Top Token Logprobs Over Time - {variant_description}<br>Prompt: {prompt_preview}",
        xaxis_title="Iteration",
        yaxis_title="Log Probability",
        template="plotly_white",
    )

    fig.write_html(prompt_dir / filename)


def prompt_rows_dict_to_completion_sample(
    prompt_rows_dict: dict[str, list[CompressedOutputRow]], tokenizer: PreTrainedTokenizerBase
) -> CompletionSample:
    prompts_list = []
    completions_list = []
    for i, prompt in enumerate(prompt_rows_dict.keys()):
        rows = prompt_rows_dict[prompt]
        for row in rows:
            prompts_list.append(i)
            completions_list.append(row.text[0])
    prompts = torch.tensor(prompts_list).to(device=config.analysis.device, dtype=torch.int32)
    completions = tokenizer(
        completions_list,
        padding="max_length",
        truncation=True,
        max_length=config.sampling.gao2025.max_tokens,
        return_tensors="pt",
    )["input_ids"].to(device=config.analysis.device, dtype=torch.int32)
    return CompletionSample(prompts=prompts, completions=completions)


def gao2025_two_sample_test(
    rows_subset: dict[str, list[CompressedOutputRow]],
    unchanged_rows_subset: dict[str, list[CompressedOutputRow]],
    tokenizer: PreTrainedTokenizerBase,
    pvalue_b: int = 1000,
):
    sample1 = prompt_rows_dict_to_completion_sample(rows_subset, tokenizer)
    sample2 = prompt_rows_dict_to_completion_sample(unchanged_rows_subset, tokenizer)
    return run_two_sample_test_torch(sample1, sample2, b=pvalue_b)


def evaluate_detector_on_variant(
    source: DataSource,
    rows1: dict[str, list[CompressedOutputRow]],
    rows2: dict[str, list[CompressedOutputRow]],
    same: bool,
    prompt_length: dict[str, int],
    tokenizer: PreTrainedTokenizerBase | None = None,
    logprob_prompt: str | None = None,
    n_tests: int = 2000,
    pvalue_b: int = 1000,
) -> TwoSampleMultiTestResult:
    """
    Args:
      rows1, rows2: dictionary of prompts to lists of rows
    """
    if logprob_prompt:
        logger.info(f"Evaluating prompt {repr(logprob_prompt)}...")
    else:
        logger.info(f"Evaluating detector {source}...")
    match source:
        case DataSource.GAO2025:
            samples_per_prompt = config.sampling.gao2025.n_samples_per_prompt
            two_sample_test_fn = gao2025_two_sample_test
        case DataSource.MMLU:
            samples_per_prompt = config.sampling.mmlu.n_samples_per_prompt
            two_sample_test_fn = mmlu_two_sample_test
        case DataSource.US:
            samples_per_prompt = config.sampling.logprob.n_samples_per_prompt
            two_sample_test_fn = logprob_two_sample_test
            # Only keep the provided or default prompt, discarding the others
            if logprob_prompt is None:
                logprob_prompt = config.sampling.logprob.default_prompt
            rows1 = {logprob_prompt: rows1[logprob_prompt]}
            rows2 = {logprob_prompt: rows2[logprob_prompt]}

    token_count = {"i": [], "o": []}
    stats = []
    pvalues = []
    for _ in range(n_tests):
        if same:
            assert rows1 == rows2
            subset = {p: random.sample(rows, 2 * samples_per_prompt) for p, rows in rows1.items()}
            sample1 = {p: subset[p][:samples_per_prompt] for p in rows1.keys()}
            sample2 = {p: subset[p][samples_per_prompt:] for p in rows2.keys()}
        else:
            sample1 = {p: random.sample(rows, samples_per_prompt) for p, rows in rows1.items()}
            sample2 = {p: random.sample(rows, samples_per_prompt) for p, rows in rows2.items()}

        result = two_sample_test_fn(sample1, sample2, pvalue_b=pvalue_b, tokenizer=tokenizer)
        stats.append(result.statistic)
        if pvalue_b > 0:
            pvalues.append(result.pvalue)
        token_count["i"].append(
            sum(prompt_length[p] * len(r) for p, r in sample1.items())
            + sum(prompt_length[p] * len(r) for p, r in sample2.items())
        )
        token_count["o"].append(
            sum(r.text[1] for rows in sample1.values() for r in rows)
            + sum(r.text[1] for rows in sample2.values() for r in rows)
        )

    return TwoSampleMultiTestResult(
        stats=stats,
        pvalues=pvalues if pvalue_b > 0 else None,
        input_token_avg=sum(token_count["i"]) / len(token_count["i"]),
        output_token_avg=sum(token_count["o"]) / len(token_count["o"]),
    )


class PlotData(BaseModel):
    experiment: Literal["baseline", "ablation_prompt"]
    model_name: str
    variant: str | None
    roc_curves: dict[DataSource | str, list[tuple[list[float], list[float]]]]
    roc_auc_ci: dict[DataSource | str, CIResult]

    @field_validator("roc_curves", "roc_auc_ci", mode="before")
    @classmethod
    def validate_roc_curves(cls, v, info):
        if info.data.get("experiment") == "baseline":
            if isinstance(v, dict) and v:
                first_key = next(iter(v))
                if isinstance(first_key, str):
                    return {DataSource(int(k)): v for k, v in v.items()}
        return v


def get_plot_dir(sampling_directory: Path) -> Path:
    directory = config.plots_dir / "roc_curves" / sampling_directory.name
    os.makedirs(directory, exist_ok=True)
    return directory


def get_ablation_prompt_data_dir(sampling_directory: Path) -> Path:
    directory = get_plot_dir(sampling_directory) / "ablation_prompt" / "data"
    os.makedirs(directory, exist_ok=True)
    return directory


def get_baselines_data_dir(sampling_directory: Path) -> Path:
    directory = get_plot_dir(sampling_directory) / "baselines" / "data"
    os.makedirs(directory, exist_ok=True)
    return directory


def plot_roc_curve_with_fs_cache(plot_data: PlotData, out_dir: Path):
    os.makedirs(out_dir, exist_ok=True)
    variant_slug = slugify(_variant_preslug(plot_data.variant), max_length=200, hash_length=0)
    data_path = out_dir / f"{variant_slug}.json"
    with open(data_path, "w") as f:
        json.dump(plot_data.model_dump(mode="json"), f, indent=2)
    plot_roc_curve(data_path)


def _variant_preslug(variant: str | None) -> str:
    if variant is None:
        return "all"
    variant_metadata = orjson.loads(variant)
    variant_preslug = (
        variant_metadata["type"]
        + ","
        + ",".join(f"{k}={v}" for k, v in variant_metadata.items() if k != "type")
    )
    return variant_preslug


def plot_roc_curve(
    plot_data_path: Path,
):
    """From the path of a data file containing the necessary data, plot the ROC curves in the parent directory,
    with the same name except for the extension."""
    plot_data = PlotData.model_validate_json(plot_data_path.read_text())
    conditions = list(plot_data.roc_curves.keys())
    fig = go.Figure()

    # Define colors for each data source
    colors = px.colors.qualitative.Plotly

    for i, condition in enumerate(conditions):
        match plot_data.experiment:
            case "baseline":
                # Conditions are DataSource objects
                condition_name = condition.name
            case "ablation_prompt":
                # Conditions are prompts
                condition_name = repr(condition)

        color = colors[i % len(colors)]
        display_name = f"{condition_name} (AUC: {plot_data.roc_auc_ci[condition].avg:.4f})"

        for j, (fpr, tpr) in enumerate(plot_data.roc_curves[condition]):
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name=display_name,
                    line=dict(color=color),
                    showlegend=(j == 0),  # Only show legend for first curve of each condition
                    legendgroup=condition_name,  # Group all curves from same condition
                )
            )

    # Random chance diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random chance",
            line=dict(dash="dash", color="black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    variant = plot_data.variant
    title = f"ROC Curves on model {plot_data.model_name}"
    if variant:
        title += f", on variant: {_variant_preslug(variant)}"
    else:
        title += " across all variants"
    fig.update_layout(
        font_family="Spectral",
        template="plotly_white",
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    plot_dir = plot_data_path.parent.parent
    filename_base = plot_data_path.stem
    fig.write_html(plot_dir / f"{filename_base}.html")
    logger.info(f"Saved ROC curve to {plot_dir / f'{filename_base}.html'}")


class BAPlotData(BaseModel):
    token_count: dict[DataSource, tuple[float, float]]
    # ROC AUCs averaged across models, by variant and DataSource
    v_point_estimates: dict[Variant, dict[DataSource, float]]
    v_bootstrap_results: dict[Variant, dict[DataSource, list[float]]]
    # ROC AUCs averaged across variants, by model and DataSource
    m_point_estimates: dict[str, dict[DataSource, float]]
    m_bootstrap_results: dict[str, dict[DataSource, list[float]]]

    @field_validator(
        "token_count",
        "v_point_estimates",
        "v_bootstrap_results",
        "m_point_estimates",
        "m_bootstrap_results",
        mode="before",
    )
    @classmethod
    def validate_datasource_keys(cls, value):
        """Convert string keys back to DataSource enums during deserialization"""
        if isinstance(value, dict):
            result = {}
            for key, val in value.items():
                if isinstance(val, dict) and all(
                    isinstance(k, str) and k.isdigit() for k in val.keys()
                ):
                    # Nested dict with string/int DataSource keys
                    result[key] = {DataSource.from_str(ds): v for ds, v in val.items()}
                elif isinstance(key, str) and key.isdigit():
                    # Direct string/int DataSource key
                    result[DataSource.from_str(key)] = val
                else:
                    result[key] = val
            return result
        return value

    def sources(self) -> list[DataSource]:
        return list(self.token_count.keys())


class BaselineAnalysis:
    stats_filename = "baseline_analysis.json"
    plot_dir = config.plots_dir / "paper" / "baseline"
    overall_performance_path = plot_dir / "overall_performance.json"
    plot_data_path = plot_dir / "baseline.json"

    @staticmethod
    def compute_stats(
        directory: Path,
        data: CompressedOutput,
        sources: list[DataSource],
        n_tests: int = 1000,
        pvalue_b: int = 1000,
    ):
        if DataSource.GAO2025 in sources:
            tokenizer = AutoTokenizer.from_pretrained(data.model_name)
            if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
                logger.info(f"Setting pad token to eos token for {data.model_name}")
                tokenizer.pad_token = tokenizer.eos_token

        analysis_results = AnalysisResult(
            experiment="baseline",
            model_name=data.model_name,
            n_tests=n_tests,
            pvalue_b=pvalue_b,
        )

        logger.info(f"{sources=}")
        logger.info(f"{data.model_name=}")
        variants = [v for v in data.references.variants.keys() if v != TinyChange.unchanged_str()]
        prompts = list(data.references.prompts.keys())
        prompt_length = {prompt: tokens for prompt, tokens in prompts}

        logger.info("Splitting data by source...")
        data_by_source = {source: data.filter(datasource=source) for source in sources}

        logger.info("Getting rows by variant...")
        rows_by_variant = {
            source: data_by_source[source].get_rows_by_variant() for source in sources
        }
        unchanged_rows = {
            source: rows_by_variant[source][TinyChange.unchanged_str()] for source in sources
        }
        unchanged_rows_by_prompt = {
            source: data_by_source[source].get_rows_by_prompt(unchanged_rows[source])
            for source in sources
        }

        start_time = time.time()
        # Get data on the false positive rate
        logger.info("Evaluating on the original model")
        analysis_results.original = {
            source.to_str(): evaluate_detector_on_variant(
                source,
                unchanged_rows_by_prompt[source],
                unchanged_rows_by_prompt[source],
                same=True,
                prompt_length=prompt_length,
                tokenizer=tokenizer,
                n_tests=n_tests,
                pvalue_b=pvalue_b,
            )
            for source in sources
        }
        for variant_idx, variant in enumerate(variants):
            logger.info(f"Variant {variant_idx + 1}/{len(variants)}: {variant}")
            analysis_results.variants[variant] = {}
            for source in sources:
                variant_rows = rows_by_variant[source][variant]
                rows_by_prompt = data_by_source[source].get_rows_by_prompt(variant_rows)
                analysis_results.variants[variant][source.to_str()] = evaluate_detector_on_variant(
                    source,
                    unchanged_rows_by_prompt[source],
                    rows_by_prompt,
                    same=False,
                    prompt_length=prompt_length,
                    tokenizer=tokenizer,
                    n_tests=n_tests,
                    pvalue_b=pvalue_b,
                )
            with open(directory / BaselineAnalysis.stats_filename, "wb") as f:
                f.write(orjson.dumps(analysis_results.model_dump(mode="json")))

        end_time = time.time()
        logger.info(f"Time taken: {(end_time - start_time):.2f} seconds")

    @staticmethod
    def score_by_variant_one(
        analysis: AnalysisResult, sampling: bool
    ) -> dict[Variant, dict[DataSource, float]]:
        """Compute the AUC by variant and source for a given analysis (model)"""
        variants = analysis.variant_names
        sources = [DataSource.from_str(k) for k in analysis.conditions]
        scores = {}
        for variant in variants:
            scores[variant] = {}
            for source in sources:
                scores[variant][source] = analysis.auc(
                    variant=variant, condition=source.to_str(), sampling=sampling
                )
        return scores

    @staticmethod
    def score_by_variant(
        analyses: list[AnalysisResult], sampling: bool
    ) -> dict[Variant, dict[DataSource, float]]:
        """Compute the AUC by variant and source, average across the analyses (models)."""
        variants = analyses[0].variant_names
        sources = [DataSource.from_str(k) for k in analyses[0].conditions]
        scores = [BaselineAnalysis.score_by_variant_one(a, sampling) for a in analyses]
        results = {v: {s: 0 for s in sources} for v in variants}
        for v in variants:
            for s in sources:
                results[v][s] = sum(score[v][s] for score in scores) / len(scores)
        return results

    @staticmethod
    def score_by_variant_bootstrap_one(analyses: list[AnalysisResult]):
        """Single bootstrap iteration"""
        analyses_bootstrap = random.choices(analyses, k=len(analyses))
        return BaselineAnalysis.score_by_variant(analyses_bootstrap, sampling=True)

    @staticmethod
    def score_by_variant_bootstrap(
        analyses: list[AnalysisResult],
    ) -> dict[Variant, dict[DataSource, list[float]]]:
        """Bootstrap by sampling with replacement from the analyses, then from the statistics for each analysis."""
        n_bootstrap = config.analysis.n_bootstrap
        variants = analyses[0].variant_names
        sources = [DataSource.from_str(k) for k in analyses[0].conditions]
        results = {v: {s: [] for s in sources} for v in variants}

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(BaselineAnalysis.score_by_variant_bootstrap_one, analyses)
                for _ in range(n_bootstrap)
            ]
            for future in tqdm(futures, desc="bootstrap"):
                result = future.result()
                for v in variants:
                    for s in sources:
                        results[v][s].append(result[v][s])
        return results

    @staticmethod
    def score_by_model_bootstrap_one(analyses: list[AnalysisResult], variants: list[Variant]):
        """Single bootstrap iteration"""
        variant_bootstrap = random.choices(variants, k=len(variants))
        return BaselineAnalysis.score_by_model(analyses, variant_bootstrap, sampling=True)

    @staticmethod
    def score_by_model_bootstrap(analyses: list[AnalysisResult]):
        """Bootstrap by sampling with replacement from the variants, then from the statistics for each variant."""
        n_bootstrap = config.analysis.n_bootstrap
        model_names = [a.model_name for a in analyses]
        variants = analyses[0].variant_names
        sources = [DataSource.from_str(k) for k in analyses[0].conditions]
        results = {m: {s: [] for s in sources} for m in model_names}

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(BaselineAnalysis.score_by_model_bootstrap_one, analyses, variants)
                for _ in range(n_bootstrap)
            ]
            for future in tqdm(futures, desc="bootstrap"):
                result = future.result()
                for m in model_names:
                    for s in sources:
                        results[m][s].append(result[m][s])
        return results

    @staticmethod
    def score_by_model_one(
        analyses: list[AnalysisResult], variant: Variant, sampling: bool
    ) -> dict[str, dict[DataSource, float]]:
        """Compute the AUC by model and source for a given variant."""
        sources = [DataSource.from_str(k) for k in analyses[0].conditions]
        model_names = [a.model_name for a in analyses]
        scores = {}
        for analysis, model_name in zip(analyses, model_names):
            scores[model_name] = {}
            for source in sources:
                scores[model_name][source] = analysis.auc(
                    variant=variant, condition=source.to_str(), sampling=sampling
                )
        return scores

    @staticmethod
    def score_by_model(
        analyses: list[AnalysisResult], variants: list[Variant], sampling: bool
    ) -> dict[str, dict[DataSource, float]]:
        """Compute the AUC by model and source, averaged across the variants."""
        models = [a.model_name for a in analyses]
        sources = [DataSource.from_str(k) for k in analyses[0].conditions]
        scores = [
            BaselineAnalysis.score_by_model_one(analyses, variant, sampling=sampling)
            for variant in variants
        ]
        results = {model: {source: 0 for source in sources} for model in models}
        for model in models:
            for source in sources:
                results[model][source] = sum(score[model][source] for score in scores) / len(scores)
        return results

    @staticmethod
    def gen_plot_data_and_plot():
        BaselineAnalysis.gen_plot_data()
        BaselineAnalysis.print_data()
        BaselineAnalysis.plot()

    @staticmethod
    def gen_plot_data():
        sampling_dirs = [
            config.sampling_data_dir / "keep" / dirname
            for dirname in config.analysis.sampling_dirnames
        ]
        analyses = []
        for sampling_dir in sampling_dirs:
            p = Path(sampling_dir) / BaselineAnalysis.stats_filename
            with open(p) as f:
                analyses.append(AnalysisResult.model_validate(orjson.loads(f.read())))

        input_tokens = AnalysisResult.multianalysis_input_token_avg(analyses)
        output_tokens = AnalysisResult.multianalysis_output_token_avg(analyses)

        token_count = {
            DataSource.from_str(k): (input_tokens[k], output_tokens[k]) for k in input_tokens.keys()
        }
        for source in token_count.keys():
            token_count[source] = tuple(
                token_count[source][i] / (2 * source.get_config().n_samples_per_prompt)
                for i in range(2)
            )

        v_point_estimates = BaselineAnalysis.score_by_variant(analyses, sampling=False)
        v_bootstrap_results = BaselineAnalysis.score_by_variant_bootstrap(analyses)
        m_point_estimates = BaselineAnalysis.score_by_model(
            analyses, variants=analyses[0].variant_names, sampling=False
        )
        m_bootstrap_results = BaselineAnalysis.score_by_model_bootstrap(analyses)
        plot_data = BAPlotData(
            token_count=token_count,
            v_point_estimates=v_point_estimates,
            v_bootstrap_results=v_bootstrap_results,
            m_point_estimates=m_point_estimates,
            m_bootstrap_results=m_bootstrap_results,
        )
        os.makedirs(BaselineAnalysis.plot_dir, exist_ok=True)
        with open(BaselineAnalysis.plot_data_path, "wb") as f:
            f.write(orjson.dumps(plot_data.model_dump(mode="json")))

    @staticmethod
    def print_data():
        with open(BaselineAnalysis.plot_data_path, "rb") as f:
            plot_data = BAPlotData.model_validate(orjson.loads(f.read()))
        # TODO
        print(plot_data)

    @staticmethod
    def plot():
        with open(BaselineAnalysis.plot_data_path, "rb") as f:
            plot_data = BAPlotData.model_validate(orjson.loads(f.read()))
        difficulty_scales = {
            "finetune_no_lora": {
                "title": "finetuning",
                "match_fn": lambda v: v["type"] == "finetune" and v["lora"] is False,
                "scale_attr": "n_samples",
                "xaxis_title": "Number of steps of finetuning",
            },
            "finetune_lora": {
                "title": "LoRA finetuning",
                "match_fn": lambda v: v["type"] == "finetune" and v["lora"] is True,
                "scale_attr": "n_samples",
                "xaxis_title": "Number of steps of finetuning",
            },
            "random_noise": {
                "title": "random noise",
                "match_fn": lambda v: v["type"] == "random_noise",
                "scale_attr": "scale",
                "xaxis_title": "Standard deviation of the gaussian noise added to each weight",
            },
            "weight_pruning_magnitude": {
                "title": "weight pruning, selection by magnitude",
                "match_fn": lambda v: v["type"] == "weight_pruning" and v["method"] == "magnitude",
                "scale_attr": "scale",
                "xaxis_title": "Fraction of the weights to prune",
            },
            "weight_pruning_random": {
                "title": "weight pruning, random selection",
                "match_fn": lambda v: v["type"] == "weight_pruning" and v["method"] == "random",
                "scale_attr": "scale",
                "xaxis_title": "Fraction of the weights to prune",
            },
        }

        for scale_name, scale_info in difficulty_scales.items():
            BaselineAnalysis.plot_difficulty_scale(plot_data, scale_name, scale_info)
        BaselineAnalysis.compute_overall_performance(plot_data)

    @staticmethod
    def plot_difficulty_scale(plot_data: BAPlotData, scale_name: str, scale_info: dict):
        logger.info(f"Plotting difficulty scale: {scale_name}...")
        variant_description_subset = []
        for k in plot_data.v_point_estimates.keys():
            variant = orjson.loads(k)
            if scale_info["match_fn"](variant):
                variant_description_subset.append(k)
        scale_attr = scale_info["scale_attr"]
        variant_description_subset.sort(key=lambda k: orjson.loads(k)[scale_attr], reverse=True)
        xaxis_values = [orjson.loads(k)[scale_attr] for k in variant_description_subset]
        print(xaxis_values)

        fig = go.Figure()
        for source in plot_data.sources():
            y_values = [plot_data.v_point_estimates[v][source] for v in variant_description_subset]
            y_bootstrap_values = [
                plot_data.v_bootstrap_results[v][source] for v in variant_description_subset
            ]
            y_cis = [ci(b, config.analysis.results_alpha) for b in y_bootstrap_values]
            y_upper = [y_ci[1] for y_ci in y_cis]
            y_lower = [y_ci[0] for y_ci in y_cis]
            fig.add_trace(
                go.Scatter(
                    x=xaxis_values,
                    y=y_values,
                    name=config.plotting.source_name[source.value],
                    line_color=config.plotting.color_map[source.to_str()],
                )
            )
            # https://plotly.com/python/continuous-error-bars/
            fig.add_trace(
                go.Scatter(
                    x=xaxis_values + xaxis_values[::-1],
                    y=y_upper + y_lower[::-1],
                    fill="toself",
                    fillcolor=config.plotting.color_map[source.to_str()],
                    line=dict(color="rgba(255,255,255,0)"),
                    opacity=0.2,
                    showlegend=False,
                )
            )

        # Add horizontal line for random guessing
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="gray",
            annotation_text="Random guessing",
            annotation_position="bottom right",
        )
        if xaxis_values[-1] < 1:
            xaxis_ticktext = [f"2<sup>{round(math.log(x, 2))}</sup>" for x in xaxis_values]
        else:
            xaxis_ticktext = xaxis_values

        fig.update_layout(
            font_family="Spectral",
            font_size=18,
            template="plotly_white",
            title=f"{scale_info['title']}",
            xaxis=dict(
                title=scale_info["xaxis_title"],
                type="log",
                autorange="reversed",
                tickmode="array",
                tickvals=xaxis_values,
                ticktext=xaxis_ticktext,
            ),
            yaxis_title="ROC AUC",
        )
        fig_path = BaselineAnalysis.plot_dir / f"{scale_name}.pdf"
        fig.write_image(fig_path)
        logger.info(f"Saved plot to {fig_path}")


class PromptAblation:
    stats_filename = "prompt_ablation_analysis.json"
    plot_data_path = config.plots_dir / "paper" / "prompt_ablation.json"
    plot_path = config.plots_dir / "paper" / "prompt_ablation.pdf"

    @staticmethod
    def compute_stats(
        directory: Path,
        data: CompressedOutput,
        n_tests: int = 1000,
        pvalue_b: int = 1000,
    ):
        """Test the influence of the prompt choice on detection performance for the logprob method."""
        source = DataSource.US
        filtered_data = data.filter(datasource=source)
        rows_by_variant = filtered_data.get_rows_by_variant()
        unchanged_rows = rows_by_variant[TinyChange.unchanged_str()]
        unchanged_rows_by_prompt = filtered_data.get_rows_by_prompt(unchanged_rows)

        prompt_length = {
            prompt: tokens
            for prompt, tokens in filtered_data.references.prompts.keys()
            if prompt in unchanged_rows_by_prompt
        }
        prompts = list(prompt_length.keys())
        analysis_results = AnalysisResult(
            experiment="ablation_prompt",
            model_name=data.model_name,
            n_tests=n_tests,
            pvalue_b=pvalue_b,
        )
        variants = [v for v in data.references.variants.keys() if v != TinyChange.unchanged_str()]
        analysis_results.original = {
            prompt: evaluate_detector_on_variant(
                source,
                unchanged_rows_by_prompt,
                unchanged_rows_by_prompt,
                same=True,
                prompt_length=prompt_length,
                logprob_prompt=prompt,
                n_tests=n_tests,
                pvalue_b=pvalue_b,
            )
            for prompt in prompts
        }
        for variant_idx, variant in enumerate(variants):
            logger.info(f"Variant {variant_idx + 1}/{len(variants)}: {variant}")
            rows_by_prompt = filtered_data.get_rows_by_prompt(rows_by_variant[variant])
            analysis_results.variants[variant] = {
                prompt: evaluate_detector_on_variant(
                    source,
                    unchanged_rows_by_prompt,
                    rows_by_prompt,
                    same=False,
                    prompt_length=prompt_length,
                    logprob_prompt=prompt,
                    n_tests=n_tests,
                    pvalue_b=pvalue_b,
                )
                for prompt in prompts
            }
            with open(directory / PromptAblation.stats_filename, "wb") as f:
                f.write(orjson.dumps(analysis_results.model_dump(mode="json")))

    @staticmethod
    def compute_prompt_score(analyses: list[AnalysisResult], sampling: bool) -> dict[str, float]:
        """For each prompt, return the average over analyses (models) of its AUC minus the model average."""
        prompts = list(analyses[0].original.keys())
        scores = []
        for analysis in analyses:
            scores.append(analysis.avg_auc_across_variants(sampling=sampling, centered=True))
        return {prompt: sum(score[prompt] for score in scores) / len(scores) for prompt in prompts}

    @staticmethod
    def bootstrap_iteration(analyses: list[AnalysisResult]):
        """Single bootstrap iteration"""
        analyses_bootstrap = random.choices(analyses, k=len(analyses))
        return PromptAblation.compute_prompt_score(analyses_bootstrap, sampling=True)

    @staticmethod
    def compute_prompt_score_bootstrap(analyses: list[AnalysisResult]) -> dict[str, list[float]]:
        """Bootstrap by sampling with replacement both from the analyses, then from the statistics for each analysis."""
        n_bootstrap = config.analysis.n_bootstrap
        prompts = list(analyses[0].original.keys())
        results = {prompt: [] for prompt in prompts}

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(PromptAblation.bootstrap_iteration, analyses)
                for _ in range(n_bootstrap)
            ]
            for future in tqdm(futures, desc="bootstrap"):
                result = future.result()
                for prompt, score in result.items():
                    results[prompt].append(score)

        return results

    @staticmethod
    def gen_plot_data_and_plot():
        PromptAblation.gen_plot_data()
        PromptAblation.plot()

    @staticmethod
    def gen_plot_data():
        sampling_dirs = [
            config.sampling_data_dir / "keep" / dirname
            for dirname in config.analysis.sampling_dirnames
        ]
        analyses = []
        for sampling_dir in sampling_dirs:
            p = Path(sampling_dir) / PromptAblation.stats_filename
            with open(p) as f:
                analyses.append(AnalysisResult.model_validate(orjson.loads(f.read())))

        prompt_length_avg = AnalysisResult.multianalysis_input_token_avg(analyses)
        for prompt in prompt_length_avg.keys():
            prompt_length_avg[prompt] /= 2 * config.sampling.logprob.n_samples_per_prompt

        point_estimates = PromptAblation.compute_prompt_score(analyses, sampling=False)

        bootstrap_results = PromptAblation.compute_prompt_score_bootstrap(analyses)

        all_results = {
            "prompt_length_avg": prompt_length_avg,
            "point_estimates": point_estimates,
            "bootstrap_results": bootstrap_results,
        }
        plot_data_path = PromptAblation.plot_data_path
        with open(plot_data_path, "wb") as f:
            f.write(orjson.dumps(all_results))
        logger.info(f"Prompt ablation analysis data saved to {plot_data_path}")

    @staticmethod
    def plot():
        with open(PromptAblation.plot_data_path, "rb") as f:
            all_results = orjson.loads(f.read())
        prompt_length_avg = all_results["prompt_length_avg"]
        bootstrap_results = all_results["bootstrap_results"]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        sorted_prompts = sorted(bootstrap_results.keys(), key=lambda x: prompt_length_avg[x])

        # Violin plots for AUC advantages
        for prompt in sorted_prompts:
            scores = bootstrap_results[prompt]
            fig.add_trace(
                go.Violin(
                    x=[prompt] * len(scores),
                    y=scores,
                    name=prompt,
                    box_visible=True,
                    points="all",
                    showlegend=True,
                    legend="legend",
                ),
                secondary_y=False,
            )

        # Scatter plot for prompt lengths
        fig.add_trace(
            go.Scatter(
                x=sorted_prompts,
                y=[prompt_length_avg[prompt] for prompt in sorted_prompts],
                mode="markers+text",
                name="Average Prompt Token Length Across Models",
                marker=dict(size=10, color="white", line_width=2),
                text=[f"{prompt_length_avg[prompt]:.1f}" for prompt in sorted_prompts],
                textposition="middle right",
                textfont=dict(size=14),
                showlegend=True,
                legend="legend2",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            font_family="Spectral",
            template="plotly_white",
            title="AUC Advantages of Prompts and Their Average Lengths",
            xaxis_showticklabels=False,
            legend=dict(x=0, y=1, xanchor="left", yanchor="top"),
            legend2=dict(x=0.35, y=1, xanchor="left", yanchor="top"),
        )

        fig.update_yaxes(
            title_text="Overall AUC Advantage", tickformat=".0%", dtick=0.01, secondary_y=False
        )
        fig.update_yaxes(showgrid=False, showticklabels=False, secondary_y=True)

        plot_path = PromptAblation.plot_path
        fig.write_html(plot_path)
        logger.info(f"Prompt ablation analysis plot saved to {plot_path}")


def debug_mmlu_correct_rate(debug_dir: Path):
    import sys

    from track_llm_apis.sampling.analyze_mmlu import is_correct

    debug_dir = (
        # config.sampling_data_dir / "keep" / "2025-09-12_11-16-46_Qwen2fQwen2.5-0.5B-Instruct"
        config.sampling_data_dir / "keep" / "2025-09-12_11-16-45_google2fgemma-3-1b-it"
    )
    compressed_output = CompressedOutput.from_json_dir(debug_dir)
    compressed_output = compressed_output.filter(datasource=DataSource.MMLU)
    rows_by_variant = compressed_output.get_rows_by_variant()
    correct_rates = []
    for variant, rows in rows_by_variant.items():
        print(f"Variant: {variant}")
        correct = 0
        total = 0
        for row in rows:
            total += 1
            if is_correct(row.prompt[0], row.text[0]):
                correct += 1
        print(f"  Rate of correct MMLU answers: {correct / total}")
        correct_rates.append(correct / total)
    print(f"Average rate of correct MMLU answers: {sum(correct_rates) / len(correct_rates)}")
    sys.exit()


if __name__ == "__main__":
    analysis_config = config.analysis

    if analysis_config.task == "compute_stats":
        output_dir = config.sampling_data_dir / "keep" / analysis_config.sampling_dirname
        compressed_output = CompressedOutput.from_json_dir(output_dir)
        logger.info(compressed_output.model_name)
        logger.info(f"number of rows: {len(compressed_output.rows)}")
        for ref_attr in compressed_output.references.__dict__.keys():
            ref = getattr(compressed_output.references, ref_attr)
            logger.info(f"length of field '{ref_attr}': {len(ref)}")

        if analysis_config.experiment == "baseline":
            BaselineAnalysis.compute_stats(
                directory=output_dir,
                data=compressed_output,
                sources=[DataSource.US, DataSource.MMLU, DataSource.GAO2025],
                n_tests=analysis_config.n_tests,
                pvalue_b=analysis_config.pvalue_b,
            )
        elif analysis_config.experiment == "ablation_prompt":
            PromptAblation.compute_stats(
                directory=output_dir,
                data=compressed_output,
                n_tests=analysis_config.n_tests,
                pvalue_b=analysis_config.pvalue_b,
            )

    elif analysis_config.task == "plot":
        if analysis_config.experiment == "ablation_prompt":
            PromptAblation.gen_plot_data_and_plot()
        elif analysis_config.experiment == "baseline":
            BaselineAnalysis.gen_plot_data_and_plot()
