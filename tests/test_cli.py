from __future__ import annotations

import asyncio
from pathlib import Path

from typer.testing import CliRunner

from navjev import Navigator
from navjev.cli import app
from navjev.llm import ScriptedLlmClient
from tests.helpers import ScriptedJevClient, by_title
from tests.test_build import echo_summarizer

runner = CliRunner()


def _indexed(tmp_path: Path, fixtures_dir: Path) -> tuple[Path, Path]:
    nav = Navigator(
        tmp_path / "idx",
        jev_client=ScriptedJevClient(
            by_title({"Financial Statements": 0.9, "Cash Flow": 0.95})
        ),
        llm_client=ScriptedLlmClient(echo_summarizer),
    )
    nav.add(fixtures_dir / "sample.md")
    nav.jev._record_to = tmp_path / "fixture.jsonl"
    asyncio.run(nav.a_search("capex?"))
    return tmp_path / "idx", tmp_path / "fixture.jsonl"


def test_ask_replays_a_recorded_fixture(tmp_path: Path, fixtures_dir: Path) -> None:
    index, fixture = _indexed(tmp_path, fixtures_dir)
    result = runner.invoke(
        app, ["ask", "capex?", "--index", str(index), "--detail", "--replay", str(fixture)]
    )
    assert result.exit_code == 0, result.output
    assert "-> Financial Statements" in result.output
    assert (
        "[0.95] Acme Corp Annual Report 2023 > Financial Statements > Cash Flow"
        in result.output
    )
    assert "cost $" in result.output


def test_ask_json(tmp_path: Path, fixtures_dir: Path) -> None:
    import json

    index, fixture = _indexed(tmp_path, fixtures_dir)
    result = runner.invoke(
        app, ["ask", "capex?", "-i", str(index), "--json", "--replay", str(fixture)]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["results"][0]["title"] == "Cash Flow" and not data["no_answer"]


def test_ask_on_empty_index_fails_clearly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ask", "anything?", "-i", str(tmp_path / "empty")])
    assert result.exit_code != 0
    assert "add a document first" in str(result.exception)


def test_add_skips_unstructured_files(tmp_path: Path) -> None:
    flat = tmp_path / "flat.md"
    flat.write_text("no headings\n")
    result = runner.invoke(app, ["add", str(flat), "-i", str(tmp_path / "idx")])
    assert result.exit_code == 0
    assert "skipped" in result.output
