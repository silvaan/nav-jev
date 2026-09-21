"""Arm C: the PageIndex-style policy, an LLM deciding which branches to open.

Implements the same TraversalPolicy protocol as JevPolicy and runs inside the same
BeamSearch over the same tree, so the only difference between arm C and arm D is who
answers. Anything else that differs invalidates the comparison.
"""

from __future__ import annotations

from navjev.llm import LlmClient
from navjev.traverse.beam import llm_decide
from navjev.traverse.questions import Thresholds, build_state
from navjev.types import ChildDecision, DocumentTree, Expansion, NodeId


class LlmPolicy:
    def __init__(self, llm: LlmClient, model: str, thresholds: Thresholds) -> None:
        self.llm = llm
        self.model = model
        self.thresholds = thresholds

    def reset(self) -> None:
        pass

    async def expand(self, tree: DocumentTree, node_id: NodeId, query: str) -> Expansion:
        """Asks the LLM for a relevance rating per child and a stop decision, returned as
        structured output, then maps them onto the same Expansion record so traces from
        both arms are directly comparable."""
        node = tree.nodes[node_id]
        children = tree.children_of(node_id)
        state = build_state(tree, node_id, query)
        decision = await llm_decide(self.llm, self.model, state, len(children))
        return Expansion(
            node_id=node_id,
            depth=node.depth,
            children=[
                ChildDecision(
                    child_id=child.id,
                    raw_score=d.raw_score,
                    normalized=d.normalized,
                    confidence=d.confidence,
                    probabilities=d.probabilities,
                    expanded=False,
                )
                for child, d in zip(children, decision.children, strict=True)
            ],
            stop_here=decision.stop_here,
            escalated_to_llm=False,
            input_tokens=decision.input_tokens,
            latency_ms=decision.latency_ms,
            model_id=decision.model_id,
            off_topic=None,
            cost_usd=decision.cost_usd,
        )
