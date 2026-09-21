from __future__ import annotations

import math
import re
from collections.abc import Sequence

from navjev.baselines.bm25 import Bm25Retriever, tokenize
from navjev.baselines.dense import DenseRetriever, chunk_text
from navjev.types import DocumentTree, TreeNode


def make_tree() -> DocumentTree:
    nodes = {
        "root": TreeNode("root", "Report", 0, "", children=["a", "b", "c"]),
        "a": TreeNode(
            "a", "Business", 1, "Widgets and gadgets sold across regions.", parent_id="root"
        ),
        "b": TreeNode(
            "b",
            "Cash flow",
            1,
            "Capital expenditure was 412 million in fiscal 2023.",
            parent_id="root",
        ),
        "c": TreeNode(
            "c", "Compensation", 1, "Officer pay policy and bonuses.", parent_id="root"
        ),
    }
    return DocumentTree("doc", "Report", "root", nodes, "x", "spec", "native")


class BagOfWords:
    model_name = "bag-of-words"

    def __init__(self) -> None:
        self._tokens = 0

    @property
    def tokens_used(self) -> int:
        return self._tokens

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vocab = [chr(c) for c in range(ord("a"), ord("z") + 1)]
        out = []
        for t in texts:
            self._tokens += len(t) // 4
            words = re.findall(r"\w+", t.lower())
            vec = [0.0] * 26
            for w in words:
                if w[0] in vocab:
                    vec[vocab.index(w[0])] += 1.0
            # Add a hash bucket per whole word so "capital" and "cash" differ.
            vec.extend(0.0 for _ in range(64))
            for w in words:
                vec[26 + hash(w) % 64] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


class KeywordReranker:
    model_name = "keyword"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) / (len(q) or 1) for t in texts]


def test_bm25_ranks_the_section_with_the_terms() -> None:
    r = Bm25Retriever(make_tree())
    result = r.retrieve("capital expenditure fiscal 2023", max_nodes=2)
    assert result.node_ids[0] == "b"
    assert result.cost_usd == 0.0 and result.latency_ms >= 0
    assert result.trace.expansions == []


def test_bm25_returns_nothing_for_unrelated_query() -> None:
    assert Bm25Retriever(make_tree()).retrieve("quantum chromodynamics").nodes == []


def test_chunking_overlaps_and_covers() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_chars=10, chunk_overlap=4)
    assert chunks[0] == "abcdefghij" and chunks[1].startswith("ghij")
    assert "".join(c[-6:] for c in chunks).endswith("z")
    assert chunk_text("short", 10, 2) == ["short"]


def test_dense_with_reranker_maps_chunks_back_to_nodes() -> None:
    tree = make_tree()
    backend = BagOfWords()
    r = DenseRetriever(tree, backend, KeywordReranker(), chunk_chars=30, chunk_overlap=5)
    assert len(r.chunks) > len(tree.nodes)  # long sections were split
    assert r.index_embedding_tokens > 0
    result = r.retrieve("capital expenditure fiscal 2023", max_nodes=2)
    assert result.node_ids[0] == "b"
    assert len(set(result.node_ids)) == len(result.node_ids)
    assert result.query_tokens > 0


def test_dense_without_reranker_uses_cosine() -> None:
    result = DenseRetriever(make_tree(), BagOfWords(), None).retrieve("officer pay bonuses")
    assert result.node_ids[0] == "c"
