from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from navjev.build.index import Index
from navjev.build.parsers import LlmStructureParser, MarkdownParser, PdfParser
from navjev.build.summarize import (
    MAX_SUMMARY_WORDS,
    SUMMARY_PROMPT_VERSION,
    Summarizer,
    SummaryCache,
    clip_words,
    summary_cache_key,
)
from navjev.llm import LlmRate, ScriptedLlmClient
from navjev.types import TreeNode


def echo_summarizer(model: str, system: str, user: str, schema: Any) -> dict[str, Any]:
    title = next(line for line in user.splitlines() if line.startswith("Section title: "))
    return {"summary": f"Summary of {title.removeprefix('Section title: ')}."}


async def test_summarizer_fills_every_node_and_uses_cache(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    tree = MarkdownParser().parse(fixtures_dir / "sample.md")
    llm = ScriptedLlmClient(echo_summarizer, rates={"m": LlmRate(1.0, 5.0)})
    cache = SummaryCache(tmp_path / "cache")
    s = Summarizer("m", cache, llm)
    await s.summarize_tree(tree)
    assert all(n.summary for n in tree.nodes.values())
    assert tree.nodes["root"].summary == "Summary of Acme Corp Annual Report 2023."
    first = s.last_report
    assert first.llm_calls == len(tree.nodes) and first.cache_hits == 0
    assert first.cost_usd > 0

    # Second pass over the same tree is free.
    tree2 = MarkdownParser().parse(fixtures_dir / "sample.md")
    calls_before = len(llm.calls)
    await Summarizer("m", cache, llm).summarize_tree(tree2)
    assert len(llm.calls) == calls_before
    assert tree2.nodes["root"].summary == tree.nodes["root"].summary


def test_cache_key_changes_with_model_and_prompt_version() -> None:
    node = TreeNode("x", "T", 0, "body")
    k = summary_cache_key(node, "m")
    assert k != summary_cache_key(node, "m2")
    assert k != summary_cache_key(node, "m", "v2")
    assert k == summary_cache_key(TreeNode("y", "T", 4, "body"), "m", SUMMARY_PROMPT_VERSION)


def test_clip_words_enforces_the_cap() -> None:
    text = " ".join(["w"] * (MAX_SUMMARY_WORDS + 10))
    clipped, truncated = clip_words(text, MAX_SUMMARY_WORDS)
    assert truncated and len(clipped.split()) == MAX_SUMMARY_WORDS
    assert clip_words("short", MAX_SUMMARY_WORDS) == ("short", False)


async def test_summary_prompt_carries_ancestors_and_subsections(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    tree = MarkdownParser().parse(fixtures_dir / "sample.md")
    llm = ScriptedLlmClient(echo_summarizer)
    await Summarizer("m", SummaryCache(tmp_path), llm).summarize_tree(tree)
    cash = next(c for c in llm.calls if "Section title: Cash Flow" in c["user"])
    assert "Acme Corp Annual Report 2023 > Financial Statements" in cash["user"]
    fin = next(c for c in llm.calls if "Section title: Financial Statements" in c["user"])
    assert "Subsections: Cash Flow; Balance Sheet" in fin["user"]


def test_index_build_and_reload(tmp_path: Path, fixtures_dir: Path) -> None:
    llm = ScriptedLlmClient(echo_summarizer, rates={"m": LlmRate(1.0, 5.0)})
    index = Index.build(fixtures_dir / "sample.md", tmp_path / "idx", "m", llm)
    assert index.doc_ids() == ["sample"]
    assert (tmp_path / "idx" / "docs" / "sample.json").exists()

    reopened = Index(tmp_path / "idx")
    tree = reopened.load_tree("sample")
    assert tree.nodes["root"].summary
    cash = next(n for n in tree.nodes.values() if n.title == "Cash Flow")
    assert "$412 million" in reopened.node_text("sample", cash.id)
    m = reopened.manifest()
    assert m["document_count"] == 1
    assert m["llm_inferred_structure_count"] == 0
    assert m["summarizer_models"] == ["m"]
    assert m["indexing_cost_usd"] > 0
    docs = m["documents"]
    assert isinstance(docs, dict)
    assert docs["sample"]["summarization"]["prompt_version"] == SUMMARY_PROMPT_VERSION
    with pytest.raises(KeyError):
        reopened.load_tree("missing")


def test_llm_structure_parser_marks_tree_inferred(tmp_path: Path) -> None:
    def outline(model: str, system: str, user: str, schema: Any) -> dict[str, Any]:
        return {
            "headings": [
                {"line": 1, "level": 1, "title": "Introduction"},
                {"line": 4, "level": 1, "title": "Results"},
                {"line": 6, "level": 2, "title": "Details"},
            ]
        }

    p = tmp_path / "flat.txt"
    p.write_text("Introduction\nsome text\nmore\nResults\nresult text\nDetails\ndetail text\n")
    parser = LlmStructureParser(ScriptedLlmClient(outline), model="m")
    tree = parser.parse(p)
    assert tree.structure_source == "llm_inferred"
    assert tree.parser == "text+llm_structure"
    titles = {n.title: n for n in tree.nodes.values()}
    assert titles["Details"].parent_id == titles["Results"].id
    assert titles["Results"].text == "result text"
    assert titles["Introduction"].text == "some text\nmore"


def test_pdf_fallback_records_page_spans(tmp_path: Path) -> None:
    import fitz

    def outline(model: str, system: str, user: str, schema: Any) -> dict[str, Any]:
        lines = [ln for ln in user.splitlines() if ": " in ln and ln.split(":")[0].isdigit()]
        found = []
        for ln in lines:
            num, _, content = ln.partition(": ")
            if content.strip() in ("Alpha", "Beta"):
                found.append({"line": int(num), "level": 1, "title": content.strip()})
        return {"headings": found}

    doc = fitz.open()
    for text in ["Alpha\nalpha body", "Beta\nbeta body"]:
        doc.new_page().insert_text((72, 72), text)
    p = tmp_path / "flat.pdf"
    doc.save(str(p))
    tree = PdfParser(LlmStructureParser(ScriptedLlmClient(outline), "m")).parse(p)
    assert tree.structure_source == "llm_inferred"
    titles = {n.title: n for n in tree.nodes.values()}
    assert titles["Alpha"].page_span == (1, 1)
    assert titles["Beta"].page_span == (2, 2)
    assert json.loads(tree.to_json())["structure_source"] == "llm_inferred"
