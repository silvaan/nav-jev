"""Frozen splits with recorded hashes. The runner refuses an unfrozen split.

A split is a directory:

    data/<dataset>/<split>/queries.json    the EvalQuery records
    data/<dataset>/<split>/documents/      one file per document
    data/<dataset>/<split>/frozen.json     content hash and provenance

`content_hash` covers the queries and the bytes of every document, so a split whose
documents were re-downloaded or edited no longer matches and is refused.

Gold sections are resolved against the index at run time, from the evidence the dataset
provides: page numbers (FinanceBench), evidence paragraphs (QASPER), or passage titles
(NanoHotpotQA). The resolution is in `resolve_gold` and is the same for every arm.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from navjev.types import DocumentTree, NodeId


@dataclass
class EvalQuery:
    query_id: str
    doc_id: str
    question: str
    answer: str | list[str]
    gold_node_ids: list[str] | None = None
    """Section-level ground truth where the dataset provides span annotations. None means
    only answer-level metrics are available for this query."""
    evidence_pages: list[int] | None = None
    evidence_texts: list[str] | None = None
    evidence_titles: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalQuery:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Split:
    name: str
    queries: list[EvalQuery]
    documents: dict[str, Path]
    content_hash: str
    frozen: bool
    dataset: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_ids(self) -> list[str]:
        return sorted(self.documents)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(queries: Sequence[EvalQuery], documents: dict[str, Path]) -> str:
    h = hashlib.sha256()
    payload = json.dumps([q.to_dict() for q in queries], sort_keys=True, ensure_ascii=False)
    h.update(payload.encode("utf-8"))
    for doc_id in sorted(documents):
        h.update(doc_id.encode("utf-8"))
        h.update(_sha256_file(documents[doc_id]).encode("ascii"))
    return h.hexdigest()


def split_dir(data_dir: Path, dataset: str, split: str) -> Path:
    return Path(data_dir) / dataset / split


def freeze_split(
    dataset: str,
    split: str,
    queries: Sequence[EvalQuery],
    documents: dict[str, Path],
    data_dir: Path,
    provenance: dict[str, Any] | None = None,
) -> Split:
    """Copy documents in, write queries, record the hash. Refuses to overwrite an existing
    frozen split, since silently refreezing is how a hash stops meaning anything."""
    target = split_dir(data_dir, dataset, split)
    if (target / "frozen.json").exists():
        raise FileExistsError(f"{target} is already frozen; delete it deliberately to refreeze")
    docs_dir = target / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for doc_id, src in documents.items():
        dst = docs_dir / f"{doc_id}{Path(src).suffix}"
        if Path(src).resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        copied[doc_id] = dst
    (target / "queries.json").write_text(
        json.dumps([q.to_dict() for q in queries], indent=1, ensure_ascii=False), "utf-8"
    )
    digest = content_hash(queries, copied)
    (target / "frozen.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "split": split,
                "content_hash": digest,
                "query_count": len(queries),
                "document_count": len(copied),
                "provenance": provenance or {},
            },
            indent=2,
        ),
        "utf-8",
    )
    return Split(split, list(queries), copied, digest, True, dataset, provenance or {})


def load_split(dataset: str, split: str, data_dir: Path) -> Split:
    """Read a split back and re-verify its hash. `frozen` is True only if the recorded
    hash matches the bytes on disk now."""
    target = split_dir(data_dir, dataset, split)
    queries_path = target / "queries.json"
    if not queries_path.exists():
        raise FileNotFoundError(f"no split at {target}; run `nav-jev dataset freeze {dataset}`")
    queries = [EvalQuery.from_dict(q) for q in json.loads(queries_path.read_text("utf-8"))]
    docs_dir = target / "documents"
    documents = {p.stem: p for p in sorted(docs_dir.iterdir())} if docs_dir.exists() else {}
    digest = content_hash(queries, documents)
    frozen_path = target / "frozen.json"
    recorded: dict[str, Any] = (
        json.loads(frozen_path.read_text("utf-8")) if frozen_path.exists() else {}
    )
    frozen = recorded.get("content_hash") == digest
    return Split(
        split, queries, documents, digest, frozen, dataset, recorded.get("provenance", {})
    )


def assign_split(query_id: str, dev_fraction: float = 0.2) -> str:
    """Deterministic dev/test assignment by hash of the query id."""
    bucket = int(hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if bucket < dev_fraction else "test"


# --------------------------------------------------------------------------------------
# Gold resolution against a tree
# --------------------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def resolve_gold(query: EvalQuery, tree: DocumentTree) -> list[NodeId] | None:
    """Section-level gold for a query, from whatever evidence the dataset gives.

    Pages: every node whose page span covers an evidence page. Texts: the deepest node
    whose own text contains the evidence paragraph. Titles: the node with that title.
    Returns None when the dataset gives no evidence, so recall is skipped honestly.
    """
    if query.gold_node_ids is not None:
        return list(query.gold_node_ids)
    gold: list[NodeId] = []

    if query.evidence_pages:
        for node in tree.walk():
            if node.page_span is None:
                continue
            lo, hi = node.page_span
            if any(lo <= p <= hi for p in query.evidence_pages) and node.id not in gold:
                gold.append(node.id)

    if query.evidence_texts:
        for evidence in query.evidence_texts:
            needle = _norm(evidence)
            if not needle:
                continue
            best: NodeId | None = None
            best_depth = -1
            for node in tree.walk():
                if needle in _norm(node.text) and node.depth > best_depth:
                    best, best_depth = node.id, node.depth
            if best is None:
                # Fall back to a prefix match: parsers may cut a paragraph across nodes.
                head = needle[:120]
                for node in tree.walk():
                    if head in _norm(node.text) and node.depth > best_depth:
                        best, best_depth = node.id, node.depth
            if best is not None and best not in gold:
                gold.append(best)

    if query.evidence_titles:
        wanted = {_norm(t) for t in query.evidence_titles}
        for node in tree.walk():
            if _norm(node.title) in wanted and node.id not in gold:
                gold.append(node.id)

    if not (query.evidence_pages or query.evidence_texts or query.evidence_titles):
        return None
    return gold


# --------------------------------------------------------------------------------------
# Dataset builders. Network and the `datasets` package are needed only here.
# --------------------------------------------------------------------------------------


def _hf_dataset(name: str, config: str | None = None, split: str | None = None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install the `eval` extra: uv sync --extra eval") from error
    return (
        load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    )


def _download(url: str, dest: Path) -> Path:
    import httpx

    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)
    return dest


def _freeze_both(
    dataset: str,
    queries: Iterable[EvalQuery],
    documents: dict[str, Path],
    data_dir: Path,
    provenance: dict[str, Any],
    max_queries: int | None,
) -> dict[str, Split]:
    by_split: dict[str, list[EvalQuery]] = {"dev": [], "test": []}
    for q in queries:
        by_split[assign_split(q.query_id)].append(q)
    out: dict[str, Split] = {}
    for split_name, qs in by_split.items():
        qs = sorted(qs, key=lambda q: q.query_id)
        if max_queries is not None:
            qs = qs[:max_queries]
        docs = {q.doc_id: documents[q.doc_id] for q in qs if q.doc_id in documents}
        out[split_name] = freeze_split(dataset, split_name, qs, docs, data_dir, provenance)
    return out


FINANCEBENCH_PDF_MIRROR = "https://github.com/patronus-ai/financebench/raw/main/pdfs"


def load_financebench(
    cache_dir: Path, data_dir: Path = Path("data"), max_queries: int | None = None
) -> dict[str, Split]:
    """Long single-document QA over filings. The setting tree retrieval claims to win.

    Evidence is given as page numbers, which become section gold through page spans.
    Filings are fetched from the URLs the dataset records; each PDF's hash enters the
    split hash, so a changed upload is a changed split.
    """
    rows = _hf_dataset("PatronusAI/financebench", split="train")
    documents: dict[str, Path] = {}
    queries: list[EvalQuery] = []
    for row in rows:
        doc_id = str(row["doc_name"])
        if doc_id not in documents:
            dest = cache_dir / "financebench" / f"{doc_id}.pdf"
            # The authors' repository mirrors every filing; investor-relations links rot.
            mirror = f"{FINANCEBENCH_PDF_MIRROR}/{doc_id}.pdf"
            try:
                documents[doc_id] = _download(mirror, dest)
            except Exception:  # noqa: BLE001 - any failure falls back to the original link
                documents[doc_id] = _download(str(row["doc_link"]), dest)
        pages = sorted(
            {
                int(e["evidence_page_num"]) + 1
                for e in row.get("evidence", [])
                if e.get("evidence_page_num") is not None
            }
        )
        texts = [
            str(e["evidence_text"]) for e in row.get("evidence", []) if e.get("evidence_text")
        ]
        queries.append(
            EvalQuery(
                query_id=str(row["financebench_id"]),
                doc_id=doc_id,
                question=str(row["question"]),
                answer=str(row["answer"]),
                evidence_pages=pages or None,
                evidence_texts=texts or None,
            )
        )
    provenance = {"source": "hf:PatronusAI/financebench", "split": "train"}
    return _freeze_both("financebench", queries, documents, data_dir, provenance, max_queries)


def qasper_paper_markdown(row: dict[str, Any]) -> str:
    """A paper's section structure as Markdown, so the native parser recovers it. Accepts
    the archive layout (`full_text` is a list of sections) and the HF layout (a dict of
    parallel lists)."""
    lines = [f"# {row['title']}", "", str(row.get("abstract", "")).strip(), ""]
    full = row["full_text"]
    if isinstance(full, dict):
        sections = list(zip(full["section_name"], full["paragraphs"], strict=True))
    else:
        sections = [(sec["section_name"], sec["paragraphs"]) for sec in full]
    for name, paragraphs in sections:
        heading = str(name).strip() or "(untitled section)"
        level = 2 + heading.count(":::")  # QASPER nests with ':::'
        heading = heading.split(":::")[-1].strip() or heading
        lines.append(f"{'#' * min(level, 6)} {heading}")
        lines.append("")
        for para in paragraphs:
            lines.append(str(para).strip())
            lines.append("")
    return "\n".join(lines)


QASPER_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
QASPER_FILES = {"dev": "qasper-train-v0.3.json", "test": "qasper-dev-v0.3.json"}


def load_qasper(
    cache_dir: Path, data_dir: Path = Path("data"), max_queries: int | None = None
) -> dict[str, Split]:
    """Scientific papers with paragraph-level answer spans, which gives section recall a
    ground truth rather than a proxy.

    Read from the authors' v0.3 archive (the HF copy is a loading script, which current
    `datasets` refuses). Our dev split is QASPER train and our test split is QASPER dev,
    so threshold fitting and evaluation never share a paper. Questions every annotator
    marked unanswerable are dropped; the rest keep every annotator's answer as gold.
    """
    import tarfile

    archive = _download(QASPER_URL, cache_dir / "qasper" / "qasper-train-dev-v0.3.tgz")
    with tarfile.open(archive) as tar:
        tar.extractall(cache_dir / "qasper", filter="data")
    documents: dict[str, Path] = {}
    by_split: dict[str, list[EvalQuery]] = {}
    for split_name, filename in QASPER_FILES.items():
        papers = json.loads((cache_dir / "qasper" / filename).read_text("utf-8"))
        queries: list[EvalQuery] = []
        for doc_id, paper in papers.items():
            path = cache_dir / "qasper" / "papers" / f"{doc_id}.md"
            if doc_id not in documents:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(qasper_paper_markdown(paper), "utf-8")
                documents[doc_id] = path
            for qi, qa in enumerate(paper["qas"]):
                golds: list[str] = []
                evidence: list[str] = []
                for entry in qa["answers"]:
                    a = entry["answer"]
                    if a.get("unanswerable"):
                        continue
                    if a.get("yes_no") is not None:
                        golds.append("yes" if a["yes_no"] else "no")
                    elif a.get("free_form_answer"):
                        golds.append(str(a["free_form_answer"]))
                    elif a.get("extractive_spans"):
                        golds.extend(str(x) for x in a["extractive_spans"])
                    evidence.extend(str(e) for e in a.get("evidence", []) if e)
                if not golds:
                    continue
                queries.append(
                    EvalQuery(
                        query_id=f"{doc_id}:{qi:02d}",
                        doc_id=doc_id,
                        question=str(qa["question"]),
                        answer=golds,
                        evidence_texts=evidence or None,
                    )
                )
        by_split[split_name] = queries
    provenance = {"source": QASPER_URL, "files": QASPER_FILES}
    dev, test = by_split["dev"], by_split["test"]
    out: dict[str, Split] = {}
    for split_name, qs in (("dev", dev), ("test", test)):
        qs = sorted(qs, key=lambda q: q.query_id)
        if max_queries is not None:
            qs = qs[:max_queries]
        docs = {q.doc_id: documents[q.doc_id] for q in qs}
        out[split_name] = freeze_split("qasper", split_name, qs, docs, data_dir, provenance)
    return out


def passages_markdown(passages: Sequence[tuple[str, str]]) -> str:
    """One tree over many passages, bucketed by title initial then by first two letters.

    This is the multi-hop stress case: the buckets are uninformative on purpose, so the
    summarizer and the beam have to do the work, and a query needing two passages needs
    two branches.
    """

    def key(title: str, n: int) -> str:
        cleaned = re.sub(r"[^a-z0-9]", "", title.lower())
        return (cleaned[:n] or "#").ljust(n, "#")

    ordered = sorted(passages, key=lambda p: (key(p[0], 2), p[0]))
    lines: list[str] = ["# Passage collection", ""]
    last1 = last2 = None
    for title, text in ordered:
        k1, k2 = key(title, 1), key(title, 2)
        if k1 != last1:
            lines += [f"## Titles starting with {k1.upper()}", ""]
            last1, last2 = k1, None
        if k2 != last2:
            lines += [f"### Titles starting with {k2.upper()}", ""]
            last2 = k2
        lines += [f"#### {title.strip() or '(untitled)'}", "", text.strip(), ""]
    return "\n".join(lines)


def load_nanohotpotqa(
    cache_dir: Path, data_dir: Path = Path("data"), max_queries: int | None = None
) -> dict[str, Split]:
    """Multi-hop stress case: a single path cannot succeed, so the beam has to carry two
    branches or the arm fails honestly."""
    corpus = _hf_dataset("zeta-alpha-ai/NanoHotpotQA", "corpus", split="train")
    qs = _hf_dataset("zeta-alpha-ai/NanoHotpotQA", "queries", split="train")
    qrels = _hf_dataset("zeta-alpha-ai/NanoHotpotQA", "qrels", split="train")
    titles: dict[str, str] = {}
    passages: list[tuple[str, str]] = []
    for row in corpus:
        cid = str(row["_id"])
        title = str(row.get("title") or cid)
        titles[cid] = title
        passages.append((title, str(row["text"])))
    doc_id = "nanohotpotqa"
    path = cache_dir / "nanohotpotqa" / f"{doc_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(passages_markdown(passages), "utf-8")

    relevant: dict[str, list[str]] = {}
    for row in qrels:
        relevant.setdefault(str(row["query-id"]), []).append(
            titles.get(str(row["corpus-id"]), "")
        )
    queries = [
        EvalQuery(
            query_id=str(row["_id"]),
            doc_id=doc_id,
            question=str(row["text"]),
            answer=[],  # NanoBEIR ships relevance labels, not answer strings
            evidence_titles=relevant.get(str(row["_id"])) or None,
        )
        for row in qs
    ]
    provenance = {"source": "hf:zeta-alpha-ai/NanoHotpotQA"}
    return _freeze_both(
        "nanohotpotqa", queries, {doc_id: path}, data_dir, provenance, max_queries
    )


LOADERS = {
    "financebench": load_financebench,
    "qasper": load_qasper,
    "nanohotpotqa": load_nanohotpotqa,
}
