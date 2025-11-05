from datetime import datetime
from pathlib import Path

import numpy as np
import orjson
from pydantic import BaseModel

from track_llm_apis.analyze import get_db_data
from track_llm_apis.util import slugify


class LogprobVector(BaseModel, arbitrary_types_allowed=True):
    """A vector of returned logprobs and the corresponding tokens. May be returned to multiple queries if non-determinism is low."""

    tokens: list[str]
    logprobs: list[np.float32]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogprobVector):
            return False
        return self.tokens == other.tokens and self.logprobs == other.logprobs


class IdxLogprobVector(BaseModel, arbitrary_types_allowed=True):
    """Like `LogprobVector`, but the tokens are replaced by indices."""

    tokens: list[int]
    logprobs: list[np.float32]


class LogprobResponse(BaseModel):
    """A logprob vector returned to a specific query."""

    date: datetime
    logprob_vector: LogprobVector


class MonthlyData:
    """Results of queries to an LLM API, for a given prompt, in a given month."""

    logprob_filename: str = "logprobs.json"
    queries_filename: str = "queries.json"

    def __init__(self, prompt: str, responses: list[LogprobResponse]):
        self.prompt = prompt
        self.responses = responses

    @staticmethod
    def _condense_responses(
        responses: list[LogprobResponse],
    ) -> tuple[list[LogprobVector], list[tuple[datetime, int]]]:
        logprobs = []
        queries = []
        for response in responses:
            try:
                idx = logprobs.index(response.logprob_vector)
            except ValueError:
                logprobs.append(response.logprob_vector)
                idx = len(logprobs) - 1
            queries.append((response.date.strftime("%d %H:%M:%S"), idx))
        return logprobs, queries

    @staticmethod
    def _condense_logprobs(
        logprobs: list[LogprobVector],
    ) -> tuple[list[str], list[IdxLogprobVector]]:
        tokens = []
        idx_logprobs = []
        for logprob in logprobs:
            token_indices = []
            for token in logprob.tokens:
                try:
                    idx = tokens.index(token)
                except ValueError:
                    tokens.append(token)
                    idx = len(tokens) - 1
                token_indices.append(idx)
            idx_logprobs.append(IdxLogprobVector(tokens=token_indices, logprobs=logprob.logprobs))
        return tokens, idx_logprobs

    def serialize(self, path: Path):
        """Serialize into two JSON files: `self.logprob_filename` and `self.queries_filename`"""
        path.mkdir(parents=True, exist_ok=True)
        logprobs, queries = self._condense_responses(self.responses)
        tokens, idx_logprobs = self._condense_logprobs(logprobs)
        with open(path / self.logprob_filename, "wb") as f:
            json_dict = {
                "seen_tokens": tokens,
                "seen_logprobs": [lp.model_dump(mode="python") for lp in idx_logprobs],
            }
            f.write(orjson.dumps(json_dict, option=orjson.OPT_SERIALIZE_NUMPY))
        with open(path / self.queries_filename, "wb") as f:
            f.write(orjson.dumps(queries))


def main(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    data = get_db_data(
        prompt="x",
        tables=["openai#gpt-4o"],
        after=datetime(2025, 9, 1),
        before=datetime(2025, 10, 1),
        sort_by_date=True,
    )

    for table_name, table_data in data.items():
        table_dir = dest_dir / slugify(table_name, max_length=250, hash_length=0) / "2025"
        for response_data in table_data:
            if response_data.prompt != "x":
                continue
        responses = []
        for response_data in table_data:
            responses.append(
                LogprobResponse(
                    date=response_data.date,
                    logprob_vector=LogprobVector(
                        tokens=response_data.top_tokens,
                        logprobs=[np.float32(lp) for lp in response_data.logprobs],
                    ),
                )
            )
        months = set([response.date.month for response in responses])
        for month in months:
            monthly_data = MonthlyData(
                prompt=response_data.prompt,
                responses=[response for response in responses if response.date.month == month],
            )
            monthly_data.serialize(path=table_dir / f"{month:02d}")


if __name__ == "__main__":
    main(dest_dir=Path("/tmp/tmp-data"))
