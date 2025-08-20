import gzip
import json
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field
from vllm import RequestOutput

from track_llm_apis.util import fast_hash, slugify


class DataSource(Enum):
    US = 0
    MMLU = 1
    GAO2025 = 2


class OutputRow(BaseModel):
    variant: str
    source: DataSource
    # (prompt, input_tokens)
    prompt: tuple[str, int]
    # (text, output_tokens)
    text: tuple[str, int]
    logprobs: list[dict[int, float]] | None = None

    @classmethod
    def from_request_output(
        cls, request_output: RequestOutput, variant: str, source: DataSource
    ) -> list["OutputRow"]:
        prompt_length = (
            len(request_output.prompt_token_ids)
            if request_output.prompt_token_ids is not None
            else 0
        )
        prompt = (request_output.prompt or "", prompt_length)
        rows = []
        for output in request_output.outputs:
            if output.logprobs is None:
                logprobs_dicts = None
            else:
                logprobs_dicts = [
                    {int(k): v.logprob for k, v in logprobs.items()} for logprobs in output.logprobs
                ]
            rows.append(
                cls(
                    variant=variant,
                    source=source,
                    prompt=prompt,
                    text=(output.text, len(output.token_ids)),
                    logprobs=logprobs_dicts,
                )
            )
        return rows


class CompressedOutputRow(BaseModel):
    source: int
    variant_idx: int
    prompt_idx: int
    text_idx: int
    logprobs_idx: int | None = None

    def to_json(self) -> tuple[int, int, int, int, int | None]:
        return (self.source, self.variant_idx, self.prompt_idx, self.text_idx, self.logprobs_idx)

    @classmethod
    def from_json(cls, json_data: Sequence[int | None]) -> Self:
        return cls(**dict(zip(cls.__annotations__, json_data)))


class Reference(BaseModel):
    row_attr: str
    elems: list[Any] = Field(default_factory=list)
    hash_to_idx: dict[str, int] = Field(default_factory=dict)

    _ROW_ATTR_TO_COMPRESSED_ROW_ATTR = {
        "variant": "variant_idx",
        "prompt": "prompt_idx",
        "text": "text_idx",
        "logprobs": "logprobs_idx",
    }

    @property
    def compressed_row_attr(self) -> str:
        return self._ROW_ATTR_TO_COMPRESSED_ROW_ATTR[self.row_attr]


class CompressedOutput(BaseModel):
    model_name: str
    rows: list[CompressedOutputRow] = Field(default_factory=list)
    references: list[Reference] = Field(
        default_factory=lambda: [
            Reference(row_attr="variant"),
            Reference(row_attr="prompt"),
            Reference(row_attr="text"),
            Reference(row_attr="logprobs"),
        ]
    )

    def add_row(self, row: OutputRow):
        compressed_row_kwargs = {"source": row.source.value}
        for ref in self.references:
            elem = row.__getattribute__(ref.row_attr)
            if elem is None:
                assert ref.row_attr == "logprobs"  # only logprobs can be None
                compressed_row_kwargs[ref.compressed_row_attr] = None
                continue

            if isinstance(elem, str):
                elem_hash = fast_hash(elem)
            else:
                elem_hash = fast_hash(json.dumps(elem))
            elem_hash = fast_hash(str(elem))
            elem_idx = ref.hash_to_idx.get(elem_hash, None)
            if elem_idx is None:
                # This element isn't stored yet
                elem_idx = len(ref.elems)
                ref.elems.append(elem)
                ref.hash_to_idx[elem_hash] = elem_idx
            compressed_row_kwargs[ref.compressed_row_attr] = elem_idx

        self.rows.append(CompressedOutputRow(**compressed_row_kwargs))

    def dump_json(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        json_filename = f"{slugify(self.model_name, max_length=200, hash_length=0)}.json.gz"
        json_path = output_dir / json_filename
        json_dict = {
            "model_name": self.model_name,
            "rows": [row.to_json() for row in self.rows],
            "references": {ref.row_attr: ref.elems for ref in self.references},
        }
        with gzip.open(json_path, "wt") as f:
            json.dump(json_dict, f, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_dir: Path) -> Self:
        # Find the only .json.gz file in the directory
        json_paths = list(json_dir.glob("*.json.gz"))
        if len(json_paths) != 1:
            raise ValueError(f"Expected 1 .json.gz file in {json_dir}, got {len(json_paths)}")
        json_path = json_paths[0]
        with gzip.open(json_path, "rt") as f:
            json_dict = json.load(f)
        return cls(
            model_name=json_dict["model_name"],
            rows=[CompressedOutputRow.from_json(row) for row in json_dict["rows"]],
            references=[
                Reference(row_attr=name, elems=elems)
                for name, elems in json_dict["references"].items()
            ],
        )


class UncompressedOutput(BaseModel):
    model_name: str
    rows: list[OutputRow] = Field(default_factory=list)

    @staticmethod
    def from_compressed_output(
        compressed_output: CompressedOutput, keep_datasource: DataSource | None
    ) -> Self:
        rows = []
        _indices = {
            "variant": 0,
            "prompt": 1,
            "text": 2,
            "logprobs": 3,
        }
        for row in compressed_output.rows:
            if keep_datasource is not None and row.source != keep_datasource.value:
                continue
            variant = compressed_output.references[_indices["variant"]].elems[row.variant_idx]
            prompt = compressed_output.references[_indices["prompt"]].elems[row.prompt_idx]
            text = compressed_output.references[_indices["text"]].elems[row.text_idx]
            if row.logprobs_idx is None:
                logprobs = None
            else:
                logprobs = compressed_output.references[_indices["logprobs"]].elems[
                    row.logprobs_idx
                ]
            rows.append(
                OutputRow(
                    variant=variant, source=row.source, prompt=prompt, text=text, logprobs=logprobs
                )
            )
        return UncompressedOutput(model_name=compressed_output.model_name, rows=rows)

    def rows_by_variant(self) -> dict[str, list[OutputRow]]:
        rows_by_variant = defaultdict(list)
        for row in self.rows:
            rows_by_variant[row.variant].append(row)
        return rows_by_variant

    @staticmethod
    def rows_by_prompt(rows: list[OutputRow]) -> dict[str, list[OutputRow]]:
        rows_by_prompt = defaultdict(list)
        for row in rows:
            rows_by_prompt[row.prompt[0]].append(row)
        return dict(rows_by_prompt)
