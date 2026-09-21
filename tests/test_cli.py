from __future__ import annotations

import asyncio
from pathlib import Path

from typer.testing import CliRunner

from navjev.build.index import Index
from navjev.cli import app
from navjev.llm import ScriptedLlmClient
from navjev.traverse.beam import BeamSearch, JevPolicy
from navjev.traverse.questions import Thresholds
from tests.helpers import ScriptedJevClient, by_title
from tests.test_build import echo_summarizer
from tests.test_eval_pipeline import fitted_thresholds, make_runner, make_workspace

runner = CliRunner()


def test_ask_replays_a_recorded_fixture(tmp_path: Path, fixtures_dir: Path) -> None:
    index = Index.build(
        fixtures_dir / "sample.md", tmp_path / "idx", "m", ScriptedLlmClient(echo_summarizer)
    )
    tree = index.load_tree("sample")
    client = ScriptedJevClient(by_title({"Financial Statements": 0.9, "Cash Flow": 0.95}))
    client._record_to = tmp_path / "fixture.jsonl"
    t = Thresholds().without_fallback()
    asyncio.run(BeamSearch(JevPolicy(client, t), t).retrieve(tree, "capex?"))
    result = runner.invoke(
        app,
        ["ask", str(tmp_path / "idx"), "capex?", "--replay", str(tmp_path / "fixture.jsonl")],
    )
    assert result.exit_code == 0, result.output
    assert "-> Financial Statements" in result.output
    assert "Financial Statements > Cash Flow" in result.output
    assert "model=['jev-1.13.0']" in result.output


def test_ask_without_spend_cap_is_blocked(tmp_path: Path, fixtures_dir: Path) -> None:
    Index.build(
        fixtures_dir / "sample.md", tmp_path / "idx", "m", ScriptedLlmClient(echo_summarizer)
    )
    result = runner.invoke(app, ["ask", str(tmp_path / "idx"), "capex?"])
    assert result.exit_code != 0
    assert "max_spend_usd is zero" in str(result.exception)


def test_report_and_dataset_show(tmp_path: Path, fixtures_dir: Path) -> None:
    config = make_workspace(tmp_path, fixtures_dir)
    fitted_thresholds().save(tmp_path / "thresholds.json")
    run_dir = asyncio.run(make_runner(config).run(arms=["A", "D"]))
    result = runner.invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert "jev_traversal" in result.output and "D vs A" in result.output
    assert "calibration.csv" in result.output
    result = runner.invoke(
        app, ["dataset", "show", "toy", "--data-dir", str(tmp_path / "data")]
    )
    assert result.exit_code == 0 and "frozen=yes" in result.output


def test_bench_refusal_exits_nonzero(tmp_path: Path, fixtures_dir: Path) -> None:
    config = make_workspace(tmp_path, fixtures_dir)
    result = runner.invoke(app, ["bench", str(config), "--arms", "D"])
    assert result.exit_code == 2
    assert "refused" in result.output
