"""On-disk index: the tree as JSON plus the node text, addressable without reparsing.

Layout:

    <index>/index.json          manifest: documents, parsers, summarizer, costs, hashes
    <index>/docs/<doc_id>.json  one DocumentTree per document, summaries and text included
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from navjev.build.parsers import LlmStructureParser, parse_async
from navjev.build.summarize import (
    SUMMARY_PROMPT_VERSION,
    SummarizationReport,
    Summarizer,
    SummaryCache,
)
from navjev.llm import LlmClient
from navjev.types import DocumentTree

INDEX_FORMAT_VERSION = 1


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Index:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.docs_dir = self.path / "docs"
        self.manifest_path = self.path / "index.json"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            self._manifest: dict[str, Any] = json.loads(self.manifest_path.read_text("utf-8"))
        else:
            self._manifest = {"format_version": INDEX_FORMAT_VERSION, "documents": {}}
            self._save_manifest()
        self._trees: dict[str, DocumentTree] = {}

    # -- construction -----------------------------------------------------------------

    @classmethod
    def build(
        cls,
        doc_path: Path,
        out_dir: Path,
        summarizer_model: str,
        llm: LlmClient,
        cache_dir: Path | None = None,
        structure_fallback: LlmStructureParser | None = None,
    ) -> Index:
        index = cls(out_dir)
        cache = SummaryCache(cache_dir or out_dir / ".summary-cache")
        summarizer = Summarizer(summarizer_model, cache, llm)
        asyncio.run(index.add_document(doc_path, summarizer, structure_fallback))
        return index

    async def add_document(
        self,
        doc_path: Path,
        summarizer: Summarizer,
        structure_fallback: LlmStructureParser | None = None,
        doc_id: str | None = None,
    ) -> DocumentTree:
        tree = await parse_async(doc_path, structure_fallback)
        if doc_id is not None:
            tree.doc_id = doc_id
        await summarizer.summarize_tree(tree)
        self.add_tree(
            tree,
            summarizer.last_report,
            source_sha256=file_sha256(doc_path),
            structure_cost_usd=structure_fallback.cost_usd if structure_fallback else 0.0,
        )
        return tree

    def add_tree(
        self,
        tree: DocumentTree,
        report: SummarizationReport | None = None,
        source_sha256: str | None = None,
        structure_cost_usd: float = 0.0,
    ) -> None:
        (self.docs_dir / f"{tree.doc_id}.json").write_text(tree.to_json(), encoding="utf-8")
        self._trees[tree.doc_id] = tree
        entry: dict[str, Any] = {
            "title": tree.title,
            "source_path": tree.source_path,
            "source_sha256": source_sha256,
            "parser": tree.parser,
            "structure_source": tree.structure_source,
            "node_count": len(tree.nodes),
            "max_depth": tree.max_depth(),
            "leaf_count": len(tree.leaves()),
            "structure_cost_usd": structure_cost_usd,
            "summarization": report.to_dict() if report else None,
        }
        self._manifest["documents"][tree.doc_id] = entry
        self._manifest["summary_prompt_version"] = SUMMARY_PROMPT_VERSION
        self._save_manifest()

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self._manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -- access -----------------------------------------------------------------------

    def doc_ids(self) -> list[str]:
        return sorted(self._manifest["documents"])

    def has(self, doc_id: str) -> bool:
        return doc_id in self._manifest["documents"]

    def load_tree(self, doc_id: str) -> DocumentTree:
        if doc_id not in self._trees:
            path = self.docs_dir / f"{doc_id}.json"
            if not path.exists():
                raise KeyError(f"{doc_id} is not in the index at {self.path}")
            self._trees[doc_id] = DocumentTree.from_json(path.read_text("utf-8"))
        return self._trees[doc_id]

    def node_text(self, doc_id: str, node_id: str) -> str:
        """Full text for a selected node, fetched only after retrieval."""
        return self.load_tree(doc_id).nodes[node_id].text

    def manifest(self) -> dict[str, object]:
        """Parser, structure source, node count, summarizer model and cost, hashes."""
        docs = self._manifest["documents"]
        total_cost = sum(
            (d.get("summarization") or {}).get("cost_usd", 0.0)
            + d.get("structure_cost_usd", 0.0)
            for d in docs.values()
        )
        models = sorted(
            {
                str((d.get("summarization") or {}).get("model"))
                for d in docs.values()
                if d.get("summarization")
            }
        )
        return {
            "path": str(self.path),
            "format_version": self._manifest.get("format_version"),
            "summary_prompt_version": self._manifest.get("summary_prompt_version"),
            "document_count": len(docs),
            "llm_inferred_structure_count": sum(
                1 for d in docs.values() if d.get("structure_source") == "llm_inferred"
            ),
            "summarizer_models": models,
            "indexing_cost_usd": total_cost,
            "documents": docs,
        }
