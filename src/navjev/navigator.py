"""The one class most users need.

    nav = Navigator("index/")
    nav.add("report.pdf")
    hits = nav.search("What was the 2023 capex?")

`add` parses a document into its section tree and writes one summary per section with a
text LLM (cached, so unchanged documents cost nothing twice). `search` walks the tree
with Jev and returns the sections most likely to hold the answer, best first, with the
path to each and what the query cost. Every method has an `a_` async twin.

Keys come from the environment or a `.env` file in the working directory:
`TYPESAFE_API_KEY` for Jev, and `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for the
summarizer, chosen by the model's name (`gpt-*` or `claude-*`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping
from pathlib import Path
from typing import Any, Self, TypeVar

from navjev.build.index import Index
from navjev.build.parsers import LlmStructureParser
from navjev.build.summarize import Summarizer, SummaryCache
from navjev.env import load_dotenv
from navjev.jev import JevClient
from navjev.llm import LlmClient, LlmRate
from navjev.traverse.beam import BeamSearch, JevPolicy
from navjev.traverse.questions import DEFAULT_THRESHOLDS, Thresholds
from navjev.types import DocumentTree, RetrievalResult

T = TypeVar("T")


def _run(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError("an event loop is already running; use the a_ async methods")


class Navigator:
    def __init__(
        self,
        index_dir: str | Path = "index",
        *,
        summarizer: str = "gpt-4.1-mini",
        jev_model: str = "jev-latest",
        max_spend_usd: float | None = None,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
        allow_llm_structure: bool = False,
        cache_dir: str | Path | None = None,
        jev_rate_per_million: float = 0.042,
        llm_rates: Mapping[str, LlmRate] | None = None,
        jev_client: JevClient | None = None,
        llm_client: LlmClient | None = None,
    ) -> None:
        """
        index_dir: where trees and summaries are stored; created if missing.
        summarizer: model that writes section summaries at `add` time.
        max_spend_usd: optional cap on what this Navigator may spend, per client.
        thresholds: the beam's decision thresholds; see `traverse/questions.py`.
        allow_llm_structure: let an LLM infer an outline for files that have none. The
            resulting tree is tagged `llm_inferred` in the index.
        jev_rate_per_million: Jev's price, only used to report `cost_usd`.
        llm_rates: per-model (input, output) prices per million tokens for the
            summarizer, only used for `usage`; an unpriced model is listed as such.
        """
        load_dotenv()
        self.index = Index(Path(index_dir))
        self.thresholds = thresholds.without_fallback()
        self.jev = jev_client or JevClient(
            model=jev_model, max_spend_usd=max_spend_usd, rate_per_million=jev_rate_per_million
        )
        self.llm = llm_client or LlmClient(rates=llm_rates, max_spend_usd=max_spend_usd)
        self._summarizer = Summarizer(
            summarizer,
            SummaryCache(Path(cache_dir) if cache_dir else self.index.path / "cache"),
            self.llm,
        )
        self._structure_fallback = (
            LlmStructureParser(self.llm, summarizer) if allow_llm_structure else None
        )
        self._search = BeamSearch(JevPolicy(self.jev, self.thresholds), self.thresholds)

    # -- documents --------------------------------------------------------------------

    @property
    def docs(self) -> list[str]:
        """Ids of the indexed documents (the file stem unless `add` was given one)."""
        return self.index.doc_ids()

    def tree(self, doc: str) -> DocumentTree:
        return self.index.load_tree(doc)

    async def a_add(
        self, path: str | Path, doc: str | None = None, replace: bool = False
    ) -> str:
        """Index a document. Returns its id. Already-indexed ids are skipped unless
        `replace` is set; unchanged sections hit the summary cache either way."""
        path = Path(path)
        doc_id = doc or path.stem
        if self.index.has(doc_id) and not replace:
            return doc_id
        await self.index.add_document(
            path, self._summarizer, self._structure_fallback, doc_id=doc_id
        )
        return doc_id

    def add(self, path: str | Path, doc: str | None = None, replace: bool = False) -> str:
        return _run(self.a_add(path, doc, replace))

    # -- search -----------------------------------------------------------------------

    async def a_search(
        self, query: str, doc: str | None = None, top_k: int = 5, detail: bool = False
    ) -> dict[str, Any]:
        """Find the sections most likely to answer `query`.

        Searches one document, or every document in the index when `doc` is None, each
        in its own walk, and merges the hits by score. Returns::

            {"results": [{"doc", "node_id", "title", "path", "pages", "score", "text"}],
             "no_answer": bool,      # every searched document judged the query off-topic
             "cost_usd": float,
             "traces": [...]}        # only with detail=True: every decision, per document

        A section's `score` is the probability that admitted it: the child score from its
        parent's expansion for leaves, or its own stop probability for a section that
        was judged to answer the query itself. Use it to rank, not as a calibrated
        magnitude.
        """
        doc_ids = [doc] if doc else self.docs
        if not doc_ids:
            raise ValueError("the index is empty; add a document first")
        trees = [self.index.load_tree(d) for d in doc_ids]
        results: list[RetrievalResult] = list(
            await asyncio.gather(*(self._search.retrieve(t, query, top_k) for t in trees))
        )
        hits: list[dict[str, Any]] = []
        for tree, result in zip(trees, results, strict=True):
            for node, score in zip(result.nodes, result.scores, strict=True):
                hits.append(
                    {
                        "doc": tree.doc_id,
                        "node_id": node.id,
                        "title": node.title,
                        "path": tree.path_to(node.id),
                        "pages": list(node.page_span) if node.page_span else None,
                        "score": score,
                        "text": node.text,
                    }
                )
        hits.sort(key=lambda h: -h["score"])
        out: dict[str, Any] = {
            "results": hits[:top_k],
            "no_answer": all(r.no_answer for r in results),
            "cost_usd": sum(r.cost_usd for r in results),
        }
        if detail:
            out["traces"] = [
                {"doc": t.doc_id, "text": r.trace.as_path_text(t), **r.trace.to_dict()}
                for t, r in zip(trees, results, strict=True)
            ]
        return out

    def search(
        self, query: str, doc: str | None = None, top_k: int = 5, detail: bool = False
    ) -> dict[str, Any]:
        return _run(self.a_search(query, doc, top_k, detail))

    # -- accounting and cleanup -------------------------------------------------------

    @property
    def usage(self) -> dict[str, Any]:
        """What this Navigator has spent so far, by client."""
        return {
            "jev": {
                "requests": self.jev.usage.requests,
                "input_tokens": self.jev.usage.input_tokens,
                "spent_usd": self.jev.spent_usd,
                "model_ids": list(self.jev.usage.model_ids),
            },
            "llm": {
                "requests": self.llm.usage.requests,
                "input_tokens": self.llm.usage.input_tokens,
                "output_tokens": self.llm.usage.output_tokens,
                "spent_usd": self.llm.spent_usd,
                "unpriced_models": list(self.llm.usage.unpriced_models),
            },
        }

    async def aclose(self) -> None:
        await self.jev.aclose()
        await self.llm.aclose()

    def close(self) -> None:
        _run(self.aclose())

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
