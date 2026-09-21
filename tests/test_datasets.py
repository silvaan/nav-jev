from __future__ import annotations

import json
from pathlib import Path

import pytest

from navjev.build.parsers import MarkdownParser
from navjev.eval.datasets import (
    EvalQuery,
    assign_split,
    freeze_split,
    load_split,
    passages_markdown,
    qasper_paper_markdown,
    resolve_gold,
)


def test_freeze_and_load_verify_hash(tmp_path: Path, fixtures_dir: Path) -> None:
    q = EvalQuery("q1", "sample", "capex?", "412 million", evidence_texts=["$412 million"])
    split = freeze_split("toy", "test", [q], {"sample": fixtures_dir / "sample.md"}, tmp_path)
    assert split.frozen
    back = load_split("toy", "test", tmp_path)
    assert back.frozen and back.content_hash == split.content_hash
    assert back.queries[0].evidence_texts == ["$412 million"]
    # Editing a document breaks the freeze.
    doc = back.documents["sample"]
    doc.write_text(doc.read_text() + "\n\nextra")
    assert not load_split("toy", "test", tmp_path).frozen
    with pytest.raises(FileExistsError):
        freeze_split("toy", "test", [q], {"sample": fixtures_dir / "sample.md"}, tmp_path)


def test_assign_split_is_deterministic_and_roughly_20_percent() -> None:
    ids = [f"q{i}" for i in range(2000)]
    dev = sum(assign_split(i) == "dev" for i in ids)
    assert 300 < dev < 500
    assert assign_split("q1") == assign_split("q1")


def test_resolve_gold_from_pages_texts_titles(fixtures_dir: Path) -> None:
    tree = MarkdownParser().parse(fixtures_dir / "sample.md")
    cash = next(n for n in tree.nodes.values() if n.title == "Cash Flow")
    fin = next(n for n in tree.nodes.values() if n.title == "Financial Statements")
    q = EvalQuery(
        "q", "sample", "?", "x", evidence_texts=["capital expenditure was $412 million"]
    )
    assert resolve_gold(q, tree) == [cash.id]
    q = EvalQuery("q", "sample", "?", "x", evidence_titles=["financial statements"])
    assert resolve_gold(q, tree) == [fin.id]
    assert resolve_gold(EvalQuery("q", "sample", "?", "x"), tree) is None
    assert resolve_gold(EvalQuery("q", "s", "?", "x", gold_node_ids=["n0001"]), tree) == [
        "n0001"
    ]
    for n in tree.nodes.values():
        n.page_span = None
    cash.page_span = (4, 4)
    fin.page_span = (3, 5)
    q = EvalQuery("q", "sample", "?", "x", evidence_pages=[4])
    assert set(resolve_gold(q, tree) or []) == {cash.id, fin.id}


def test_qasper_markdown_recovers_native_structure() -> None:
    row = {
        "title": "A Paper",
        "abstract": "We study things.",
        "full_text": [
            {"section_name": "Introduction", "paragraphs": ["Intro para."]},
            {"section_name": "Method ::: Data", "paragraphs": ["Data para."]},
            {"section_name": "Method ::: Model", "paragraphs": ["Model para."]},
        ],
    }
    md = qasper_paper_markdown(row)
    tree = MarkdownParser().parse_text(md, "p", "A Paper")
    assert tree.title == "A Paper"
    titles = [n.title for n in tree.walk()]
    assert titles == ["A Paper", "Introduction", "Data", "Model"]
    q = EvalQuery("q", "p", "?", "x", evidence_texts=["Model para."])
    assert resolve_gold(q, tree) == [next(n.id for n in tree.walk() if n.title == "Model")]


def test_passages_markdown_buckets_titles() -> None:
    md = passages_markdown([("Alpha", "a text"), ("Alps", "b text"), ("Beta", "c text")])
    tree = MarkdownParser().parse_text(md, "c", "Passage collection")
    depths = {n.title: n.depth for n in tree.walk()}
    assert depths["Alpha"] == 3 and depths["Beta"] == 3
    assert depths["Titles starting with AL"] == 2
    q = EvalQuery("q", "c", "?", [], evidence_titles=["Alpha", "Beta"])
    assert len(resolve_gold(q, tree) or []) == 2
    assert json.loads(tree.to_json())["structure_source"] == "native"
