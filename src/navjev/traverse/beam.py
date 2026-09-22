"""Beam search over a DocumentTree, with Jev as the traversal policy.

The loop is ordinary code. The only thing the model does is answer, per node, how likely
each child is to hold the evidence and whether the current node already suffices. Every
threshold, every cap, and every tie-break lives in `questions.py`; this module applies
them.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from navjev.jev import JevClient
from navjev.llm import LlmClient
from navjev.traverse.questions import (
    CHILD_QUESTION_PREFIX,
    LLM_TRAVERSAL_SYSTEM,
    OFF_TOPIC_QUESTION_ID,
    RELEVANCE_LEVELS,
    STOP_QUESTION_ID,
    Thresholds,
    build_state,
    expansion_questions,
    is_off_topic,
    llm_traversal_prompt,
    llm_traversal_schema,
    needs_escalation,
    select_children,
    should_stop,
)
from navjev.types import (
    ChildDecision,
    DocumentTree,
    Expansion,
    NodeId,
    RetrievalResult,
    Trace,
    TreeNode,
)


class TraversalBudgetExceeded(RuntimeError):
    """The per-query expansion budget would be exceeded. Raised, never truncated."""


class TraversalPolicy(Protocol):
    """What the beam needs from a decision-maker. JevPolicy is the one that ships; the
    protocol exists so another model can be dropped in over the same tree and beam."""

    async def expand(self, tree: DocumentTree, node_id: NodeId, query: str) -> Expansion: ...

    def reset(self) -> None:
        """Called once at the start of every query, for per-query counters."""
        ...


@dataclass
class BeamEntry:
    node_id: NodeId
    score: float
    depth: int


class BeamSearch:
    """Level-synchronous beam search.

    One level of the beam is expanded per round, with all of that level's requests issued
    concurrently, because the latency claim only holds if sibling expansions overlap.

    Dead ends: when no child of a node clears `tau_expand` and the node is not itself a
    result, the node is marked a dead end and its parent's unexpanded siblings become
    eligible again. Without this, one bad decision at depth two loses the document.

    Leaves are never expanded: a leaf has no children to score, so the score that
    admitted it to the beam is its result score. The same holds for nodes at
    `max_depth`. Results are ranked by admission score, deeper first on ties, and cut
    to `max_nodes`.
    """

    def __init__(self, policy: TraversalPolicy, thresholds: Thresholds) -> None:
        self.policy = policy
        self.thresholds = thresholds

    async def retrieve(
        self, tree: DocumentTree, query: str, max_nodes: int = 5
    ) -> RetrievalResult:
        """Walk the tree and return the selected nodes with the full trace.

        Halts on an empty beam, `max_depth`, or `max_expansions`. Returns
        `no_answer=True` when the root off-topic test clears `tau_off_topic`.
        """
        started = time.perf_counter()
        t = self.thresholds
        self.policy.reset()
        trace = Trace(query=query, doc_id=tree.doc_id)
        results: dict[NodeId, float] = {}
        expanded: dict[NodeId, Expansion] = {}
        visited: set[NodeId] = {tree.root_id}
        beam: list[BeamEntry] = [BeamEntry(tree.root_id, 1.0, 0)]
        no_answer = False

        def emit(node_id: NodeId, score: float) -> None:
            results[node_id] = max(results.get(node_id, 0.0), score)

        while beam and not no_answer:
            to_expand: list[BeamEntry] = []
            for entry in beam:
                node = tree.nodes[entry.node_id]
                if node.is_leaf or entry.depth >= t.max_depth:
                    emit(entry.node_id, entry.score)
                else:
                    to_expand.append(entry)
            if not to_expand:
                break
            if len(trace.expansions) + len(to_expand) > t.max_expansions:
                raise TraversalBudgetExceeded(
                    f"expanding {len(to_expand)} more nodes would exceed max_expansions="
                    f"{t.max_expansions} (already {len(trace.expansions)}) for query "
                    f"{query!r} on {tree.doc_id}"
                )

            expansions = await asyncio.gather(
                *(self.policy.expand(tree, e.node_id, query) for e in to_expand)
            )

            candidates: list[BeamEntry] = []
            for entry, exp in zip(to_expand, expansions, strict=True):
                trace.expansions.append(exp)
                expanded[exp.node_id] = exp
                if entry.depth == 0 and is_off_topic(exp.off_topic, t):
                    no_answer = True
                    break
                stopped = should_stop(exp.stop_here, t)
                if stopped:
                    exp.emitted = True
                    emit(exp.node_id, exp.stop_here)
                selected = select_children(exp.children, t)
                for i in selected:
                    child = exp.children[i]
                    child.expanded = True
                    candidates.append(
                        BeamEntry(child.child_id, child.normalized, entry.depth + 1)
                    )
                if not selected and not stopped:
                    exp.dead_end = True
                    candidates.extend(
                        self._rescue_siblings(tree, exp.node_id, expanded, visited)
                    )

            if no_answer:
                break
            beam = self._next_beam(candidates, visited, expanded, tree)

        ranked = sorted(results.items(), key=lambda kv: (-kv[1], -tree.nodes[kv[0]].depth))
        chosen = ranked[:max_nodes]
        nodes: list[TreeNode] = [tree.nodes[nid] for nid, _ in chosen]
        return RetrievalResult(
            nodes=nodes,
            trace=trace,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            cost_usd=trace.total_cost_usd,
            no_answer=no_answer,
            scores=[s for _, s in chosen],
        )

    @staticmethod
    def _rescue_siblings(
        tree: DocumentTree,
        node_id: NodeId,
        expanded: dict[NodeId, Expansion],
        visited: set[NodeId],
    ) -> list[BeamEntry]:
        """The parent's children that were never placed in a beam, scored as the parent's
        expansion scored them. They compete for the next beam on that score."""
        parent_id = tree.nodes[node_id].parent_id
        if parent_id is None or parent_id not in expanded:
            return []
        parent_exp = expanded[parent_id]
        depth = tree.nodes[node_id].depth
        return [
            BeamEntry(c.child_id, c.normalized, depth)
            for c in parent_exp.children
            if c.child_id not in visited
        ]

    def _next_beam(
        self,
        candidates: Sequence[BeamEntry],
        visited: set[NodeId],
        expanded: dict[NodeId, Expansion],
        tree: DocumentTree,
    ) -> list[BeamEntry]:
        """Top `beam_width` unvisited candidates by score. A rescued sibling that makes
        the cut is marked `expanded` on its parent's record so the trace shows it."""
        best: dict[NodeId, BeamEntry] = {}
        for c in candidates:
            if c.node_id in visited:
                continue
            if c.node_id not in best or c.score > best[c.node_id].score:
                best[c.node_id] = c
        ordered = sorted(best.values(), key=lambda e: (-e.score, e.depth))
        chosen = ordered[: self.thresholds.beam_width]
        for entry in chosen:
            visited.add(entry.node_id)
            parent_id = tree.nodes[entry.node_id].parent_id
            if parent_id in expanded:
                for decision in expanded[parent_id].children:
                    if decision.child_id == entry.node_id:
                        decision.expanded = True
        return chosen


# --------------------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------------------


class JevPolicy:
    """Expansion by one Jev request per node: one Score per child, plus the stop test,
    plus the off-topic test at the root, all sharing one state.

    Sends only titles and summaries. Full node text is fetched after retrieval, for the
    selected nodes only.
    """

    def __init__(
        self,
        client: JevClient,
        thresholds: Thresholds,
        llm_fallback: LlmFallback | None = None,
    ) -> None:
        self.client = client
        self.thresholds = thresholds
        self.llm_fallback = llm_fallback
        self.escalations_this_query = 0
        if llm_fallback is None and thresholds.fallback_enabled:
            raise ValueError(
                "tau_llm > 0 enables the fallback, but no LlmFallback was given; pass one or "
                "use thresholds.without_fallback()"
            )

    def reset(self) -> None:
        self.escalations_this_query = 0

    def _build_state(self, tree: DocumentTree, node_id: NodeId, query: str) -> dict[str, Any]:
        return build_state(tree, node_id, query)

    async def expand(self, tree: DocumentTree, node_id: NodeId, query: str) -> Expansion:
        node = tree.nodes[node_id]
        children = tree.children_of(node_id)
        at_root = node.parent_id is None
        state = self._build_state(tree, node_id, query)
        questions = expansion_questions(len(children), at_root)
        response = await self.client.ask(state, questions)

        decisions: list[ChildDecision] = []
        for i, child in enumerate(children):
            answer = response.answers[f"{CHILD_QUESTION_PREFIX}{i}"]
            decisions.append(
                ChildDecision(
                    child_id=child.id,
                    raw_score=float(answer.value),
                    normalized=answer.normalized(RELEVANCE_LEVELS),
                    confidence=float(answer.confidence or 0.0),
                    probabilities=dict(answer.probabilities or {}),
                    expanded=False,
                )
            )
        stop_here = response.answers[STOP_QUESTION_ID].probability
        off_topic = response.answers[OFF_TOPIC_QUESTION_ID].probability if at_root else None
        expansion = Expansion(
            node_id=node_id,
            depth=node.depth,
            children=decisions,
            stop_here=stop_here,
            escalated_to_llm=False,
            input_tokens=response.input_tokens,
            latency_ms=response.latency_ms,
            model_id=response.model,
            off_topic=off_topic,
            cost_usd=response.cost_usd(self.client.rate_per_million),
        )

        if (
            self.llm_fallback is not None
            and needs_escalation(decisions, self.thresholds)
            and self.escalations_this_query < self.llm_fallback.max_escalations_per_query
        ):
            self.escalations_this_query += 1
            redecided = await self.llm_fallback.redecide(state, len(children))
            expansion.jev_children = decisions
            expansion.jev_stop_here = stop_here
            expansion.children = [
                ChildDecision(
                    child_id=child.id,
                    raw_score=d.raw_score,
                    normalized=d.normalized,
                    confidence=d.confidence,
                    probabilities=d.probabilities,
                    expanded=False,
                )
                for child, d in zip(children, redecided.children, strict=True)
            ]
            expansion.stop_here = redecided.stop_here
            expansion.escalated_to_llm = True
            expansion.input_tokens += redecided.input_tokens
            expansion.latency_ms += redecided.latency_ms
            expansion.cost_usd += redecided.cost_usd
            expansion.model_id = f"{response.model}+{redecided.model_id}"
        return expansion


@dataclass
class LlmDecision:
    """An LLM's answer to the same questions Jev gets, mapped onto the same shape.

    The LLM returns an integer level per child, so each child decision is a point
    estimate: one-hot probabilities and confidence 1.0, marked in the trace through
    `escalated_to_llm`.
    """

    children: list[ChildDecision]
    stop_here: float
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    cost_usd: float


async def llm_decide(
    llm: LlmClient, model: str, state: dict[str, Any], child_count: int
) -> LlmDecision:
    """One LLM call deciding an expansion from the Jev state, for the fallback."""
    completion = await llm.complete_json(
        model,
        LLM_TRAVERSAL_SYSTEM,
        llm_traversal_prompt(state),
        llm_traversal_schema(child_count),
        max_tokens=1024,
    )
    levels_raw = completion.data.get("children", [])
    if len(levels_raw) != child_count:
        raise ValueError(
            f"LLM returned {len(levels_raw)} child levels for {child_count} children"
        )
    top = RELEVANCE_LEVELS - 1
    children = []
    for level_raw in levels_raw:
        level = min(max(int(level_raw), 0), top)
        children.append(
            ChildDecision(
                child_id="",
                raw_score=float(level),
                normalized=level / top,
                confidence=1.0,
                probabilities={str(level): 1.0},
                expanded=False,
            )
        )
    stop = min(max(float(completion.data.get("stop_here", 0.0)), 0.0), 1.0)
    return LlmDecision(
        children=children,
        stop_here=stop,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        latency_ms=completion.latency_ms,
        model_id=completion.model,
        cost_usd=completion.cost_usd,
    )


class LlmFallback:
    """Re-decides an expansion whose confidence fell below `tau_llm`.

    Receives the same state. Watch the escalation rate in the traces: a policy that
    escalates most of the time has not replaced the LLM, it has added a call in front
    of one.
    """

    def __init__(self, llm: LlmClient, model: str, max_escalations_per_query: int = 3) -> None:
        self.llm = llm
        self.model = model
        self.max_escalations_per_query = max_escalations_per_query

    async def redecide(self, state: dict[str, Any], child_count: int) -> LlmDecision:
        return await llm_decide(self.llm, self.model, state, child_count)
