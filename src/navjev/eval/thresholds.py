"""Fit thresholds on a dev split, verify on held-out, and lock them into a manifest.

A threshold fitted on the split it is reported on is not a result. The runner checks that
`fitted_on` names a different split from the one being evaluated.

Method. Fitting replays recorded traversals rather than re-querying Jev per grid point:

1. The dev split is traversed once with *exploration* thresholds: a low `tau_expand`,
   the widest beam under consideration, no fallback and a generous expansion budget, so
   the recorded expansions are a superset of what any candidate configuration would open.
2. Every candidate `(tau_expand, tau_stop)` is then simulated by running the real
   `BeamSearch` with a `ReplayPolicy` that serves the recorded Jev answers. Same code,
   same tie-breaks, no second implementation of the policy.
3. The cheapest candidate (fewest expansions per query) whose mean section recall holds
   at or above the target wins. If none does, the best-recall candidate is returned and
   `target_met` is False, which the manifest records.

A replay can still miss a node that a candidate wants to open and exploration did not
(the dead-end rescue can reach a low-scored sibling). Such a node is treated as a leaf
and the miss is counted; the count is in the fit report, not hidden.

`tau_llm` cannot be fitted by replay, since escalating changes the answers. It is set to
the confidence quantile that would escalate a stated fraction of dev expansions.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from navjev.eval.datasets import EvalQuery, Split, resolve_gold
from navjev.eval.metrics import section_recall
from navjev.traverse.beam import BeamSearch
from navjev.traverse.questions import (
    DEFAULT_THRESHOLDS,
    PROMPT_VERSION,
    Thresholds,
    select_children,
)
from navjev.types import DocumentTree, Expansion, NodeId, Trace

EXPAND_GRID = tuple(round(x, 2) for x in np.arange(0.30, 0.86, 0.05))
STOP_GRID = tuple(round(x, 2) for x in np.arange(0.50, 0.96, 0.05))


class ReplayMiss(LookupError):
    pass


def exploration_thresholds(
    base: Thresholds, min_tau_expand: float = min(EXPAND_GRID)
) -> Thresholds:
    """Open more than any candidate would, so replays find what they need."""
    return replace(
        base,
        tau_expand=min_tau_expand,
        tau_stop=1.01,  # never emit early; emission does not affect what gets opened
        tau_llm=0.0,
        max_expansions=max(base.max_expansions * 4, 64),
    )


class ReplayPolicy:
    """Serves recorded Jev expansions for the query currently being replayed."""

    def __init__(self) -> None:
        self._recorded: dict[NodeId, Expansion] = {}
        self.misses = 0

    def load(self, trace: Trace) -> None:
        self._recorded = {e.node_id: e for e in trace.expansions}

    def reset(self) -> None:
        pass

    async def expand(self, tree: DocumentTree, node_id: NodeId, query: str) -> Expansion:
        recorded = self._recorded.get(node_id)
        if recorded is None:
            self.misses += 1
            raise ReplayMiss(node_id)
        exp = copy.deepcopy(recorded)
        # Replay the Jev answers, not the LLM's, if the exploration run escalated.
        if exp.jev_children is not None:
            exp.children = copy.deepcopy(exp.jev_children)
            exp.jev_children = None
        if exp.jev_stop_here is not None:
            exp.stop_here = exp.jev_stop_here
            exp.jev_stop_here = None
        exp.escalated_to_llm = False
        exp.dead_end = exp.emitted = False
        for c in exp.children:
            c.expanded = False
        return exp


class _LeafOnMiss(BeamSearch):
    """A BeamSearch whose policy misses are treated as leaves, for replay only."""

    async def retrieve(self, tree: DocumentTree, query: str, max_nodes: int = 5) -> Any:
        assert isinstance(self.policy, ReplayPolicy)
        recorded_ids = set(self.policy._recorded)
        pruned = copy.deepcopy(tree)
        # A node exploration never opened has no recorded children: make it a leaf so the
        # beam emits it with its admission score instead of asking the policy.
        for node in pruned.nodes.values():
            if node.children and node.id not in recorded_ids:
                node.children = []
        return await super().retrieve(pruned, query, max_nodes)


@dataclass
class FitPoint:
    tau_expand: float
    tau_stop: float
    mean_recall: float
    mean_expansions: float
    mean_retrieved: float
    queries: int
    replay_misses: int


@dataclass
class FitResult:
    thresholds: Thresholds
    target_recall: float
    target_met: bool
    chosen: FitPoint
    grid: list[FitPoint] = field(default_factory=list)
    escalation_target: float = 0.0
    escalation_rate_at_tau_llm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": self.thresholds.to_dict(),
            "target_recall": self.target_recall,
            "target_met": self.target_met,
            "chosen": self.chosen.__dict__,
            "grid": [p.__dict__ for p in self.grid],
            "escalation_target": self.escalation_target,
            "escalation_rate_at_tau_llm": self.escalation_rate_at_tau_llm,
        }


async def simulate(
    thresholds: Thresholds,
    queries: Sequence[EvalQuery],
    traces: Mapping[str, Trace],
    trees: Mapping[str, DocumentTree],
    max_nodes: int,
) -> FitPoint:
    policy = ReplayPolicy()
    search = _LeafOnMiss(policy, thresholds)
    recalls: list[float] = []
    expansions: list[int] = []
    retrieved: list[int] = []
    for q in queries:
        trace = traces.get(q.query_id)
        if trace is None:
            continue
        tree = trees[q.doc_id]
        gold = resolve_gold(q, tree)
        if not gold:
            continue
        policy.load(trace)
        result = await search.retrieve(tree, q.question, max_nodes)
        recalls.append(section_recall(result.node_ids, gold))
        expansions.append(len(result.trace.expansions))
        retrieved.append(len(result.nodes))
    n = len(recalls)
    return FitPoint(
        thresholds.tau_expand,
        thresholds.tau_stop,
        float(np.mean(recalls)) if n else 0.0,
        float(np.mean(expansions)) if n else 0.0,
        float(np.mean(retrieved)) if n else 0.0,
        n,
        policy.misses,
    )


def decisive_confidences(traces: Mapping[str, Trace], thresholds: Thresholds) -> list[float]:
    """Minimum confidence among the decisive children of each expansion, the quantity
    `needs_escalation` compares against `tau_llm`."""
    out: list[float] = []
    for trace in traces.values():
        for e in trace.expansions:
            children = e.jev_children if e.jev_children is not None else e.children
            if not children:
                continue
            decisive = select_children(children, thresholds) or [
                max(range(len(children)), key=lambda i: children[i].normalized)
            ]
            out.append(min(children[i].confidence for i in decisive))
    return out


async def fit(
    split: Split,
    traces: Mapping[str, Trace],
    trees: Mapping[str, DocumentTree],
    target_recall: float = 0.95,
    jev_model_id: str | None = None,
    base: Thresholds = DEFAULT_THRESHOLDS,
    max_nodes: int = 5,
    expand_grid: Sequence[float] = EXPAND_GRID,
    stop_grid: Sequence[float] = STOP_GRID,
    escalation_target: float = 0.10,
) -> FitResult:
    """Sweep tau_expand and tau_stop for the cheapest configuration that holds recall at
    or above target on dev. Records the model ID and prompt version into the result."""
    if not split.frozen:
        raise ValueError(f"split {split.name} is not frozen; fit only on frozen data")
    grid: list[FitPoint] = []
    for te in expand_grid:
        for ts in stop_grid:
            candidate = replace(base, tau_expand=float(te), tau_stop=float(ts), tau_llm=0.0)
            grid.append(await simulate(candidate, split.queries, traces, trees, max_nodes))
    if not grid or all(p.queries == 0 for p in grid):
        raise ValueError("no query in the split has both a trace and resolvable gold sections")

    meeting = [p for p in grid if p.mean_recall >= target_recall]
    target_met = bool(meeting)
    pool = meeting or grid
    chosen = min(pool, key=lambda p: (p.mean_expansions, -p.mean_recall, p.tau_expand))

    confidences = decisive_confidences(traces, replace(base, tau_expand=chosen.tau_expand))
    if confidences:
        tau_llm = float(np.quantile(confidences, escalation_target))
        rate = float(np.mean([c < tau_llm for c in confidences]))
    else:
        tau_llm, rate = base.tau_llm, 0.0

    thresholds = replace(
        base,
        tau_expand=chosen.tau_expand,
        tau_stop=chosen.tau_stop,
        tau_llm=tau_llm,
        prompt_version=PROMPT_VERSION,
        jev_model_id=jev_model_id,
        fitted_on=f"{split.dataset}/{split.name}" if split.dataset else split.name,
    )
    return FitResult(
        thresholds, target_recall, target_met, chosen, grid, escalation_target, rate
    )


async def verify(
    thresholds: Thresholds,
    split: Split,
    traces: Mapping[str, Trace],
    trees: Mapping[str, DocumentTree],
    max_nodes: int = 5,
) -> dict[str, float]:
    """Recall, escalation rate, and cost proxy on a held-out split at the fitted
    thresholds, by replay of that split's exploration traces."""
    point = await simulate(
        thresholds.without_fallback(), split.queries, traces, trees, max_nodes
    )
    confidences = decisive_confidences(traces, thresholds)
    escalation = (
        float(np.mean([c < thresholds.tau_llm for c in confidences])) if confidences else 0.0
    )
    return {
        "mean_recall": point.mean_recall,
        "mean_expansions": point.mean_expansions,
        "mean_retrieved": point.mean_retrieved,
        "queries": float(point.queries),
        "replay_misses": float(point.replay_misses),
        "escalation_rate": escalation,
    }
