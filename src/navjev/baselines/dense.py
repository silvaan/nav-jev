"""Arm B: dense retrieval plus a cross-encoder reranker, the conventional pipeline.

Chunking is fixed and recorded in the config, because a dense baseline tuned worse than
practice would flatter the tree methods. State the chunk size and overlap in the manifest.

Backends are pluggable so the unit suite can run without a model download: the runner
picks `SentenceTransformerBackend` for local model names, `OpenAIEmbeddingBackend` for
`text-embedding-*` names, and `CrossEncoderReranker` for the reranker.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from navjev.baselines.bm25 import retrieval_units
from navjev.types import DocumentTree, RetrievalResult, Trace, TreeNode


class EmbeddingBackend(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    @property
    def tokens_used(self) -> int: ...


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


@dataclass
class Chunk:
    node_id: str
    text: str


def chunk_text(text: str, chunk_chars: int, chunk_overlap: int) -> list[str]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if chunk_overlap >= chunk_chars:
        raise ValueError("chunk_overlap must be smaller than chunk_chars")
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text] if text else []
    step = chunk_chars - chunk_overlap
    return [
        text[i : i + chunk_chars]
        for i in range(0, len(text), step)
        if text[i : i + chunk_chars].strip()
    ]


def chunk_tree(tree: DocumentTree, chunk_chars: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for node in retrieval_units(tree):
        for piece in chunk_text(f"{node.title}\n{node.text}", chunk_chars, chunk_overlap):
            chunks.append(Chunk(node.id, piece))
    return chunks


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class DenseRetriever:
    def __init__(
        self,
        tree: DocumentTree,
        embedding: EmbeddingBackend,
        reranker: Reranker | None = None,
        chunk_chars: int = 1200,
        chunk_overlap: int = 200,
        rerank_candidates: int = 20,
    ) -> None:
        self.tree = tree
        self.embedding = embedding
        self.reranker = reranker
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.rerank_candidates = rerank_candidates
        self.chunks = chunk_tree(tree, chunk_chars, chunk_overlap)
        started = time.perf_counter()
        before = embedding.tokens_used
        self._vectors = embedding.embed([c.text for c in self.chunks]) if self.chunks else []
        self.index_latency_ms = (time.perf_counter() - started) * 1000.0
        self.index_embedding_tokens = embedding.tokens_used - before

    @property
    def embedding_model(self) -> str:
        return self.embedding.model_name

    @property
    def reranker_model(self) -> str | None:
        return self.reranker.model_name if self.reranker else None

    def retrieve(self, query: str, max_nodes: int = 5) -> RetrievalResult:
        started = time.perf_counter()
        trace = Trace(query=query, doc_id=self.tree.doc_id)
        if not self.chunks:
            return RetrievalResult([], trace, 0.0, 0.0)
        before = self.embedding.tokens_used
        qvec = self.embedding.embed([query])[0]
        query_tokens = self.embedding.tokens_used - before
        sims = [cosine(qvec, v) for v in self._vectors]
        order = sorted(range(len(self.chunks)), key=lambda i: -sims[i])
        candidates = order[: self.rerank_candidates]
        if self.reranker is not None:
            rescored = self.reranker.score(query, [self.chunks[i].text for i in candidates])
            candidates = [
                i
                for _, i in sorted(zip(rescored, candidates, strict=True), key=lambda p: -p[0])
            ]
            final_scores = {i: s for s, i in zip(rescored, candidates, strict=False)}
        else:
            final_scores = {i: sims[i] for i in candidates}

        # Chunks map back to nodes; a node keeps its best chunk's score.
        seen: dict[str, float] = {}
        for i in candidates:
            node_id = self.chunks[i].node_id
            if node_id not in seen:
                seen[node_id] = final_scores.get(i, sims[i])
        ranked = sorted(seen.items(), key=lambda kv: -kv[1])[:max_nodes]
        nodes: list[TreeNode] = [self.tree.nodes[nid] for nid, _ in ranked]
        return RetrievalResult(
            nodes=nodes,
            trace=trace,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            cost_usd=0.0,  # priced by the runner from query_tokens and the config rate
            scores=[s for _, s in ranked],
            query_tokens=query_tokens,
        )


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------


class SentenceTransformerBackend:
    """Local embeddings through sentence-transformers (optional extra `dense`)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._tokens = 0

    @property
    def tokens_used(self) -> int:
        return self._tokens

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        self._tokens += sum(len(t) // 4 for t in texts)
        return [[float(x) for x in v] for v in vectors]


class OpenAIEmbeddingBackend:
    """OpenAI embeddings over plain HTTP. Token usage comes from the response."""

    def __init__(
        self, model_name: str, api_key: str | None = None, batch_size: int = 64
    ) -> None:
        import httpx

        self.model_name = model_name
        self._key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = httpx.Client(timeout=60.0)
        self._batch = batch_size
        self._tokens = 0

    @property
    def tokens_used(self) -> int:
        return self._tokens

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            batch = list(texts[start : start + self._batch])
            response = self._client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._key}"},
                json={"model": self.model_name, "input": batch},
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            rows = sorted(payload["data"], key=lambda d: int(d["index"]))
            out.extend([[float(x) for x in row["embedding"]] for row in rows])
            self._tokens += int(payload.get("usage", {}).get("total_tokens", 0))
        return out


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        scores = self._model.predict([(query, t) for t in texts])
        return [float(s) for s in scores]


def make_embedding_backend(model_name: str) -> EmbeddingBackend:
    if model_name.startswith("text-embedding-"):
        return OpenAIEmbeddingBackend(model_name)
    return SentenceTransformerBackend(model_name)
