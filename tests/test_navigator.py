from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from navjev import Navigator, Thresholds
from navjev.llm import LlmRate, ScriptedLlmClient
from tests.helpers import ScriptedJevClient, by_title
from tests.test_build import echo_summarizer

CAPEX = {"Financial Statements": 0.9, "Cash Flow": 0.95, "Balance Sheet": 0.3}


def make_nav(tmp_path: Path, jev_script: Any = None, **kw: Any) -> Navigator:
    return Navigator(
        tmp_path / "idx",
        jev_client=ScriptedJevClient(jev_script or by_title(CAPEX)),
        llm_client=ScriptedLlmClient(echo_summarizer, rates={"m": LlmRate(1, 5)}),
        summarizer="m",
        **kw,
    )


def test_add_then_search(tmp_path: Path, fixtures_dir: Path) -> None:
    nav = make_nav(tmp_path)
    assert nav.docs == []
    assert nav.add(fixtures_dir / "sample.md") == "sample"
    assert nav.docs == ["sample"]
    hits = nav.search("What was the fiscal 2023 capital expenditure?", top_k=2)
    assert not hits["no_answer"]
    top = hits["results"][0]
    assert top["title"] == "Cash Flow"
    assert top["path"] == ["Acme Corp Annual Report 2023", "Financial Statements", "Cash Flow"]
    assert "$412 million" in top["text"] and top["doc"] == "sample" and top["pages"] is None
    assert top["score"] == pytest.approx(0.95)
    assert hits["cost_usd"] > 0
    assert "traces" not in hits
    assert nav.usage["jev"]["requests"] == 2 and nav.usage["llm"]["spent_usd"] > 0


def test_add_is_idempotent_and_replace_reindexes(tmp_path: Path, fixtures_dir: Path) -> None:
    nav = make_nav(tmp_path)
    nav.add(fixtures_dir / "sample.md")
    calls = nav.llm.usage.requests
    nav.add(fixtures_dir / "sample.md")
    assert nav.llm.usage.requests == calls  # skipped
    nav.add(fixtures_dir / "sample.md", replace=True)
    assert nav.llm.usage.requests == calls  # re-parsed, but every summary hit the cache


def test_detail_returns_traces(tmp_path: Path, fixtures_dir: Path) -> None:
    nav = make_nav(tmp_path)
    nav.add(fixtures_dir / "sample.md")
    hits = nav.search("capex?", detail=True)
    (trace,) = hits["traces"]
    assert trace["doc"] == "sample" and "-> Financial Statements" in trace["text"]
    assert [e["node_id"] for e in trace["expansions"]] == [
        "root",
        "n0003",
    ]  # root, Financial Statements


def test_search_across_documents_merges_and_reports_no_answer(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    def script(state: dict[str, Any]) -> dict[str, Any]:
        if state["document_title"] == "Other":
            return by_title({"off_topic": 0.99})(state)
        return by_title(CAPEX)(state)

    other = tmp_path / "other.md"
    other.write_text("# Other\n\nintro\n\n## Unrelated\n\ntext\n")
    nav = make_nav(tmp_path, script)
    nav.add(fixtures_dir / "sample.md")
    nav.add(other)
    hits = nav.search("capex?")
    assert not hits["no_answer"]
    assert {h["doc"] for h in hits["results"]} == {"sample"}
    assert nav.search("capex?", doc="other")["no_answer"]


def test_empty_index_and_running_loop_errors(tmp_path: Path) -> None:
    nav = make_nav(tmp_path)
    with pytest.raises(ValueError, match="add a document"):
        nav.search("q")

    async def inside_loop() -> None:
        with pytest.raises(RuntimeError, match="a_ async"):
            nav.search("q")

    asyncio.run(inside_loop())


def test_thresholds_are_applied_without_fallback(tmp_path: Path, fixtures_dir: Path) -> None:
    nav = make_nav(tmp_path, thresholds=Thresholds(tau_expand=0.99, tau_llm=0.5))
    assert not nav.thresholds.fallback_enabled
    nav.add(fixtures_dir / "sample.md")
    assert nav.search("capex?")["results"] == []  # nothing clears 0.99


def test_context_manager_closes(tmp_path: Path) -> None:
    with make_nav(tmp_path) as nav:
        assert nav.docs == []
