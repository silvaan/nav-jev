"""Arm A: BM25 over leaf sections of the same tree.

Cheap, fast, and the honest floor. A tree-walking method that cannot beat BM25 on a
corpus is not worth its latency.
"""

from __future__ import annotations

import re
import time

from rank_bm25 import BM25Okapi

from navjev.types import DocumentTree, RetrievalResult, Trace, TreeNode

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def retrieval_units(tree: DocumentTree) -> list[TreeNode]:
    """Leaf sections, which is where the text lives. Internal nodes with their own text
    are included too, since an outline-cut PDF often keeps prose on the parent page.
    A single-node document retrieves its root."""
    units = [n for n in tree.walk() if n.is_leaf or n.text.strip()]
    return units or [tree.root]


class Bm25Retriever:
    def __init__(self, tree: DocumentTree, k1: float = 1.2, b: float = 0.75) -> None:
        self.tree = tree
        self.units = retrieval_units(tree)
        corpus = [tokenize(f"{n.title}\n{n.text}") for n in self.units]
        self._bm25 = BM25Okapi(corpus, k1=k1, b=b)

    def retrieve(self, query: str, max_nodes: int = 5) -> RetrievalResult:
        started = time.perf_counter()
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.units)), key=lambda i: -float(scores[i]))
        chosen = [i for i in order[:max_nodes] if float(scores[i]) > 0.0]
        return RetrievalResult(
            nodes=[self.units[i] for i in chosen],
            trace=Trace(query=query, doc_id=self.tree.doc_id),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            cost_usd=0.0,
            scores=[float(scores[i]) for i in chosen],
        )
