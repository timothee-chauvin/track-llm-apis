# Log Probability Tracking of LLM APIs

This is the code accompanying the paper "Log Probability Tracking of LLM APIs".

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for package management.

Run `uv sync` from the root of the repository to install its dependencies.

## Organization
The tests are in the `tests/` directory and can be run with `uv run pytest` (this requires at least one CUDA GPU to be available).

The source code lives in `src/track_llm_apis/`. The main files within this directory are:

### Remote APIs

* `logprob_prevalence_stats.py`: compute the prevalence of logprob support across OpenRouter endpoints
* `main.py`: query API endpoints and store the results
* `analyze.py`: analyze the stored logprobs

### TinyChange benchmark with local models

* `tinychange.py`: the TinyChange benchmark implementation
* `sampling/generate.py`: generate samples and store results across the TinyChange benchmark, with 3 methods: logprob tracking, MET (Model Equality Testing) and MMLU-ALG (see paper).
* `sampling/analyze.py`: analyze the stored samples
