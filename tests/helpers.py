"""Scripted clients for policy tests: answers come from a handler over the state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from navjev.jev import RecordedJevClient, State
from navjev.traverse.questions import RELEVANCE_LEVELS
from navjev.types import DocumentTree, TreeNode

# handler(state) -> {"children": [(normalized, confidence), ...], "stop": p, "off_topic": p}
Handler = Callable[[dict[str, Any]], dict[str, Any]]


class ScriptedJevClient(RecordedJevClient):
    def __init__(self, handler: Handler, model_id: str = "jev-1.13.0") -> None:
        super().__init__()
        self._handler = handler
        self._model_id = model_id

    async def _send(self, state: State, wire: Mapping[str, Any]) -> dict[str, Any]:
        assert isinstance(state, dict)
        self.calls.append({"state": dict(state), "questions": dict(wire)})
        script = self._handler(state)
        top = RELEVANCE_LEVELS - 1
        answers: dict[str, Any] = {}
        for qid, q in wire.items():
            if q["type"] == "score":
                i = int(qid.removeprefix("child_"))
                normalized, confidence = script["children"][i]
                raw = normalized * top
                lo = int(raw)
                hi = min(lo + 1, top)
                frac = raw - lo
                probs = {str(k): 0.0 for k in range(RELEVANCE_LEVELS)}
                probs[str(lo)] += 1 - frac
                probs[str(hi)] += frac
                answers[qid] = {
                    "type": "score",
                    "score": raw,
                    "legend": {str(k): f"level {k}" for k in range(RELEVANCE_LEVELS)},
                    "probabilities": probs,
                    "confidence": confidence,
                }
            elif qid == "stop_here":
                answers[qid] = {"type": "noul", "noul": script.get("stop", 0.0)}
            elif qid == "off_topic":
                answers[qid] = {"type": "noul", "noul": script.get("off_topic", 0.0)}
        return {
            "model": self._model_id,
            "answers": answers,
            "usage": {"input_tokens": 100 + 20 * len(wire), "output_tokens": 5},
        }


def tree_from_spec(spec: dict[str, Any], doc_id: str = "doc") -> DocumentTree:
    """{"title": ..., "children": [ {...}, ... ]} -> DocumentTree with ids = titles."""
    nodes: dict[str, TreeNode] = {}

    def add(item: dict[str, Any], depth: int, parent: str | None) -> str:
        title = item["title"]
        node = TreeNode(
            id=title,
            title=title,
            depth=depth,
            text=item.get("text", f"text of {title}"),
            summary=item.get("summary", f"summary of {title}"),
            parent_id=parent,
        )
        nodes[title] = node
        for child in item.get("children", []):
            node.children.append(add(child, depth + 1, title))
        return title

    root_id = add(spec, 0, None)
    return DocumentTree(doc_id, spec["title"], root_id, nodes, "spec", "spec", "native")


def by_title(scores: dict[str, tuple[float, float] | float], default: float = 0.05) -> Handler:
    """Handler that scores children by title. A bare float means confidence 0.9."""

    def handler(state: dict[str, Any]) -> dict[str, Any]:
        children = []
        for child in state["children"]:
            v = scores.get(child["title"], default)
            children.append((v, 0.9) if isinstance(v, float) else v)
        current = state["current_section"]["title"]
        stop = scores.get(f"stop@{current}", 0.0)
        off = scores.get("off_topic", 0.0)
        return {
            "children": children,
            "stop": stop if isinstance(stop, float) else stop[0],
            "off_topic": off if isinstance(off, float) else off[0],
        }

    return handler
