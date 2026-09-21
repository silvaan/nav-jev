"""The run config, validated once. Every reported number traces back to one of these."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from navjev.llm import LlmRate

ARMS = ("A", "B", "C", "D", "E")


class DatasetConfig(BaseModel):
    name: str
    split: str = "test"
    frozen: bool = True
    data_dir: str = "data"
    max_queries: int | None = None
    """Cap applied when the split is first frozen, not at run time. A frozen split is
    whatever was frozen; truncating at run time is what `--limit` does, and that marks
    the run exploratory."""


class IndexConfig(BaseModel):
    parser: str = "auto"
    summarizer_model: str
    allow_llm_inferred_structure: bool = False
    path: str | None = None
    """Defaults to index/<dataset name>."""
    cache_dir: str = ".cache/summaries"


class AnsweringConfig(BaseModel):
    model: str
    max_context_chars: int = 6000
    max_nodes: int = 5


class JevConfig(BaseModel):
    model: str = "jev-latest"
    rate_per_million: float
    max_spend_usd: float = 0.0
    max_concurrency: int = 8
    timeout_s: float = 30.0


class LlmRateConfig(BaseModel):
    input_per_million: float
    output_per_million: float


class LlmConfig(BaseModel):
    max_spend_usd: float = 0.0
    max_concurrency: int = 4
    pricing: dict[str, LlmRateConfig] = Field(default_factory=dict)

    def rates(self) -> dict[str, LlmRate]:
        return {
            m: LlmRate(r.input_per_million, r.output_per_million)
            for m, r in self.pricing.items()
        }


class ThresholdsConfig(BaseModel):
    source: str
    """Path to a thresholds JSON written by `nav-jev fit` on a dev split."""


class DenseConfig(BaseModel):
    embedding_model: str
    reranker_model: str | None = None
    chunk_chars: int = 1200
    chunk_overlap: int = 200
    rerank_candidates: int = 20
    embedding_rate_per_million: float = 0.0
    """Price per million embedding tokens. Zero for local models."""


class LlmTraversalConfig(BaseModel):
    model: str


class BaselinesConfig(BaseModel):
    dense: DenseConfig | None = None
    llm_traversal: LlmTraversalConfig | None = None


class FallbackConfig(BaseModel):
    model: str
    max_escalations_per_query: int = 3


class StatsConfig(BaseModel):
    bootstrap_resamples: int = 10_000
    seed: int = 0


class RunConfig(BaseModel):
    dataset: DatasetConfig
    index: IndexConfig
    answering: AnsweringConfig
    jev: JevConfig
    llm: LlmConfig = Field(default_factory=LlmConfig)
    thresholds: ThresholdsConfig
    arms: list[str]
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)
    fallback: FallbackConfig | None = None
    stats: StatsConfig = Field(default_factory=StatsConfig)
    results_dir: str = "results"

    raw_text: str = ""
    """The file verbatim, recorded in the manifest."""
    path: str = ""

    @field_validator("arms")
    @classmethod
    def _arms_known(cls, arms: list[str]) -> list[str]:
        unknown = [a for a in arms if a not in ARMS]
        if unknown:
            raise ValueError(f"unknown arms {unknown}; known: {ARMS}")
        return arms

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        text = Path(path).read_text(encoding="utf-8")
        data: dict[str, Any] = yaml.safe_load(text) or {}
        data["raw_text"] = text
        data["path"] = str(path)
        return cls.model_validate(data)

    def index_path(self) -> Path:
        return Path(self.index.path or f"index/{self.dataset.name}")
