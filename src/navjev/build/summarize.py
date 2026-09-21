"""Node summaries, written once per document by a cheap LLM and cached on disk.

Jev does not write these. It cannot generate text, and forcing it to by chaining choices
over characters is documented as slow and bad.

A summary is at most 60 words and is written from the node's own text plus its ancestors'
titles, so that a subsection summary still makes sense read on its own during traversal.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from navjev.llm import LlmClient
from navjev.types import DocumentTree, TreeNode

SUMMARY_PROMPT_VERSION = "v1"
MAX_SUMMARY_WORDS = 60

# A node's own text can run to hundreds of pages when an outline is coarse. The
# summarizer sees the head and the tail up to this many characters; the cap is recorded
# in the index manifest so it is a stated parameter, not a hidden one.
DEFAULT_MAX_TEXT_CHARS = 24_000

SUMMARY_SYSTEM = (
    "You write one-sentence-to-three-sentence summaries of document sections for a "
    "retrieval index. A reader will see only the section title and your summary, out of "
    "context, and must decide whether the section could contain the answer to a "
    "question. Say what the section contains: which subjects, entities, periods, figures "
    "and tables it reports, and what its subsections cover. Do not evaluate or interpret. "
    f"Use at most {MAX_SUMMARY_WORDS} words. Write in the document's language."
)

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


def summary_cache_key(
    node: TreeNode, model: str, prompt_version: str = SUMMARY_PROMPT_VERSION
) -> str:
    h = hashlib.sha256()
    h.update(node.content_hash().encode())
    h.update(b"\x00")
    h.update(model.encode())
    h.update(b"\x00")
    h.update(prompt_version.encode())
    return h.hexdigest()


class SummaryCache:
    """Keyed by sha256(node text + title) + summarizer model ID + prompt version.

    Including the model and prompt version in the key means a changed prompt is a cache
    miss rather than a tree holding summaries from two different prompts.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data.get("summary")
        return str(summary) if summary is not None else None

    def put(self, key: str, summary: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": summary}, ensure_ascii=False), encoding="utf-8")


@dataclass
class SummarizationReport:
    model: str
    prompt_version: str
    nodes: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    truncated_summaries: int = 0
    """Summaries the LLM wrote over the word cap and code cut back."""
    text_cap_chars: int = DEFAULT_MAX_TEXT_CHARS
    resolved_model_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def clip_words(text: str, max_words: int) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words), False
    return " ".join(words[:max_words]), True


def clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 3 // 4
    tail = max_chars - head
    return text[:head] + "\n[…]\n" + text[-tail:]


def summary_prompt(
    node: TreeNode, ancestors: Sequence[str], children: Sequence[TreeNode], max_text_chars: int
) -> str:
    parts = []
    if ancestors:
        parts.append("Location in the document: " + " > ".join(ancestors))
    parts.append(f"Section title: {node.title}")
    if children:
        parts.append(
            "Subsections: "
            + "; ".join(c.title for c in children[:60])
            + (" …" if len(children) > 60 else "")
        )
    body = node.text.strip()
    if body:
        parts.append(
            "Section text (own text only, excluding subsections):\n"
            + clip_text(body, max_text_chars)
        )
    else:
        parts.append(
            "The section has no text of its own; summarize what its subsections cover."
        )
    return "\n\n".join(parts)


class Summarizer:
    def __init__(
        self,
        model: str,
        cache: SummaryCache,
        llm: LlmClient,
        max_concurrency: int = 8,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    ) -> None:
        self.model = model
        self.cache = cache
        self.llm = llm
        self.max_text_chars = max_text_chars
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def summarize_tree(self, tree: DocumentTree) -> DocumentTree:
        """Fill `summary` on every node. Cost is reported and enters the amortized
        per-query indexing cost in the manifest."""
        report = SummarizationReport(
            self.model, SUMMARY_PROMPT_VERSION, text_cap_chars=self.max_text_chars
        )
        self.last_report = report
        nodes = list(tree.walk())
        report.nodes = len(nodes)
        resolved: list[str] = []
        lock = asyncio.Lock()

        async def one(node: TreeNode) -> None:
            key = summary_cache_key(node, self.model)
            cached = self.cache.get(key)
            if cached is not None:
                node.summary = cached
                async with lock:
                    report.cache_hits += 1
                return
            prompt = summary_prompt(
                node,
                [a.title for a in tree.ancestors_of(node.id)],
                tree.children_of(node.id),
                self.max_text_chars,
            )
            async with self._semaphore:
                completion = await self.llm.complete_json(
                    self.model, SUMMARY_SYSTEM, prompt, SUMMARY_SCHEMA, max_tokens=512
                )
            summary, truncated = clip_words(
                str(completion.data.get("summary", "")), MAX_SUMMARY_WORDS
            )
            node.summary = summary
            self.cache.put(key, summary)
            async with lock:
                report.llm_calls += 1
                report.input_tokens += completion.input_tokens
                report.output_tokens += completion.output_tokens
                report.cost_usd += completion.cost_usd
                report.truncated_summaries += int(truncated)
                if completion.model not in resolved:
                    resolved.append(completion.model)

        await asyncio.gather(*(one(n) for n in nodes))
        report.resolved_model_ids = resolved
        return tree
