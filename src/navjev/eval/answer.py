"""The answering step every arm shares: same model, same prompt, same character budget.

Retrieval differs between arms; nothing here does. The context is the retrieved nodes'
full text in retrieval order, cut at `max_context_chars`, and the model is asked for a
short answer or an explicit "not found".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from navjev.llm import LlmClient
from navjev.types import TreeNode

ANSWER_PROMPT_VERSION = "v1"

ANSWER_SYSTEM = (
    "You answer a question using only the document excerpts provided. Give the shortest "
    "answer that is complete: a number with its unit, a name, a short phrase, or yes/no. "
    "If the excerpts do not contain the answer, set found to false and answer to an empty "
    "string. Never guess from general knowledge."
)

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"found": {"type": "boolean"}, "answer": {"type": "string"}},
    "required": ["found", "answer"],
    "additionalProperties": False,
}

NOT_FOUND = ""


def build_context(
    nodes: Sequence[TreeNode], path_titles: Sequence[Sequence[str]], max_chars: int
) -> str:
    """Excerpts in retrieval order, each headed by its path, cut at the budget. A node
    that does not fit whole is cut, not dropped, so the budget is spent the same way for
    every arm."""
    parts: list[str] = []
    used = 0
    for node, path in zip(nodes, path_titles, strict=True):
        header = f"[{' > '.join(path)}]\n"
        body = node.text.strip()
        piece = header + body
        remaining = max_chars - used
        if remaining <= len(header):
            break
        if len(piece) > remaining:
            piece = piece[:remaining]
        parts.append(piece)
        used += len(piece) + 2
        if used >= max_chars:
            break
    return "\n\n".join(parts)


@dataclass
class AnswerResult:
    answer: str
    found: bool
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    context_chars: int


async def answer_question(
    llm: LlmClient, model: str, question: str, context: str
) -> AnswerResult:
    user = f"Question: {question}\n\nExcerpts:\n{context if context else '(no excerpts were retrieved)'}"
    completion = await llm.complete_json(
        model, ANSWER_SYSTEM, user, ANSWER_SCHEMA, max_tokens=512
    )
    found = bool(completion.data.get("found", False))
    text = str(completion.data.get("answer", "")).strip() if found else NOT_FOUND
    return AnswerResult(
        answer=text,
        found=found,
        model=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        latency_ms=completion.latency_ms,
        cost_usd=completion.cost_usd,
        context_chars=len(context),
    )
