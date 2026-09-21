"""Metrics, all of them paired across arms because every arm answers the same queries."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class QueryOutcome:
    query_id: str
    arm: str
    retrieved_node_ids: list[str]
    answer: str
    latency_ms: float
    """Retrieval wall-clock only, measured around the retriever call."""
    cost_usd: float
    """Retrieval cost for this query: Jev, LLM policy, or embedding calls."""
    input_tokens: int
    escalated: bool
    jev_model_id: str | None
    gold_node_ids: list[str] | None = None
    section_recall: float | None = None
    exact_match: float | None = None
    f1: float | None = None
    answer_cost_usd: float = 0.0
    answer_latency_ms: float = 0.0
    answer_input_tokens: int = 0
    answer_output_tokens: int = 0
    expansions: int = 0
    escalations: int = 0
    no_answer: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def total_cost_usd(self) -> float:
        return self.cost_usd + self.answer_cost_usd


def section_recall(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """The metric the traversal policy actually controls. Exact node match: a retrieved
    ancestor of a gold section does not carry the section's text, so it does not count."""
    if not gold:
        raise ValueError("section_recall needs a non-empty gold set")
    hit = set(retrieved) & set(gold)
    return len(hit) / len(set(gold))


_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    """SQuAD normalization: lowercase, strip punctuation, articles and extra whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _golds(gold: str | Sequence[str]) -> list[str]:
    return [gold] if isinstance(gold, str) else list(gold)


def exact_match(pred: str, gold: str | Sequence[str]) -> float:
    golds = _golds(gold)
    if not golds:
        raise ValueError("exact_match needs at least one gold answer")
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def token_f1(pred: str, gold: str | Sequence[str]) -> float:
    golds = _golds(gold)
    if not golds:
        raise ValueError("token_f1 needs at least one gold answer")
    p_tokens = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        g_tokens = normalize_answer(g).split()
        if not p_tokens and not g_tokens:
            best = max(best, 1.0)
            continue
        common = Counter(p_tokens) & Counter(g_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(p_tokens)
        recall = overlap / len(g_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000, seed: int = 0
) -> tuple[float, tuple[float, float]]:
    """Mean difference (a - b) and its 95% confidence interval. Report the interval, not
    only the point estimate: with a few hundred queries, one or two points of recall is
    noise."""
    if len(a) != len(b):
        raise ValueError("paired_bootstrap needs paired samples of equal length")
    if not a:
        raise ValueError("paired_bootstrap needs at least one pair")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), (float(lo), float(hi))


def calibration_curve(
    predicted: Sequence[float], outcomes: Sequence[bool], n_buckets: int = 10
) -> list[tuple[float, float, int]]:
    """Bucketed predicted probability against observed frequency.

    Returns (mean predicted, observed fraction, count) per non-empty bucket. This is the
    test of RLCD's central claim on this task. If the curve tracks the diagonal,
    thresholds transfer between documents and the method is usable. If not, every
    threshold is a per-corpus fit, and the write-up leads with that.
    """
    if len(predicted) != len(outcomes):
        raise ValueError("predicted and outcomes must be the same length")
    if n_buckets < 1:
        raise ValueError("n_buckets must be positive")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_buckets)]
    for p, o in zip(predicted, outcomes, strict=True):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"predicted probability {p} is outside [0, 1]")
        k = min(int(p * n_buckets), n_buckets - 1)
        buckets[k].append((p, o))
    out: list[tuple[float, float, int]] = []
    for items in buckets:
        if not items:
            continue
        mean_p = sum(p for p, _ in items) / len(items)
        observed = sum(1 for _, o in items if o) / len(items)
        out.append((mean_p, observed, len(items)))
    return out


def expected_calibration_error(curve: Sequence[tuple[float, float, int]]) -> float:
    total = sum(n for _, _, n in curve)
    if total == 0:
        return 0.0
    return sum(abs(p - o) * n for p, o, n in curve) / total


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))
