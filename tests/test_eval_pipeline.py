"""Threshold fitting and the runner, end to end, offline, with scripted clients."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from navjev.build.summarize import SUMMARY_SYSTEM
from navjev.eval.answer import ANSWER_SYSTEM
from navjev.eval.datasets import EvalQuery, freeze_split, load_split
from navjev.eval.report import render, write_calibration
from navjev.eval.runner import Runner, RunRefused, write_manifest
from navjev.eval.thresholds import exploration_thresholds, fit, verify
from navjev.llm import LlmRate, ScriptedLlmClient
from navjev.traverse.questions import LLM_TRAVERSAL_SYSTEM, PROMPT_VERSION, Thresholds
from tests.helpers import ScriptedJevClient, by_title
from tests.test_baselines import BagOfWords, KeywordReranker

# Jev judgments over the sample.md tree: the capex question lives under
# Financial Statements > Cash Flow; the segments question under Business > Segments.
JEV_SCRIPT = {
    "Financial Statements": 0.9,
    "Cash Flow": 0.95,
    "Balance Sheet": 0.4,
    "Business": (0.35, 0.3),
    "Segments": 0.9,
    "Executive Compensation": 0.1,
}


def jev_handler(state: dict[str, Any]) -> dict[str, Any]:
    q = state["query"]
    if "segments" in q:
        script = dict(JEV_SCRIPT, **{"Business": 0.9, "Financial Statements": 0.2})
    else:
        script = JEV_SCRIPT
    return by_title(script)(state)


def llm_handler(model: str, system: str, user: str, schema: Any) -> dict[str, Any]:
    if system == SUMMARY_SYSTEM:
        title = next(ln for ln in user.splitlines() if ln.startswith("Section title: "))
        return {"summary": f"About {title.removeprefix('Section title: ')}."}
    if system == ANSWER_SYSTEM:
        if "$412 million" in user:
            return {"found": True, "answer": "$412 million"}
        if "Widgets, Gadgets, Services" in user:
            return {"found": True, "answer": "Widgets, Gadgets, Services"}
        return {"found": False, "answer": ""}
    if system == LLM_TRAVERSAL_SYSTEM:
        state = json.loads(user.split("State:\n", 1)[1].rsplit("\n\nReturn one", 1)[0])
        good = (
            ("Financial Statements", "Cash Flow")
            if "capital" in state["query"]
            else ("Business", "Segments")
        )
        return {
            "children": [3 if c["title"] in good else 0 for c in state["children"]],
            "stop_here": 0.0,
        }
    raise AssertionError(f"unexpected system prompt: {system[:40]}")


QUERIES = {
    "dev": [
        EvalQuery(
            "d1",
            "sample",
            "What was the fiscal 2023 capital expenditure?",
            "$412 million",
            evidence_texts=["Capital expenditure was $412 million"],
        ),
        EvalQuery(
            "d2",
            "sample",
            "Which segments does Acme report?",
            ["Widgets, Gadgets, Services"],
            evidence_texts=["Widgets, Gadgets, Services."],
        ),
    ],
    "test": [
        EvalQuery(
            "t1",
            "sample",
            "What was the capital expenditure in fiscal 2023?",
            "$412 million",
            evidence_texts=["Capital expenditure was $412 million"],
        ),
        EvalQuery(
            "t2",
            "sample",
            "List the segments Acme reports.",
            ["Widgets, Gadgets, Services"],
            evidence_texts=["Widgets, Gadgets, Services."],
        ),
        EvalQuery(
            "t3",
            "sample",
            "What is the total asset figure?",
            "$9.1 billion",
            evidence_texts=["Total assets of $9.1 billion"],
        ),
    ],
}


def make_workspace(tmp_path: Path, fixtures_dir: Path, **overrides: Any) -> Path:
    for split, qs in QUERIES.items():
        freeze_split(
            "toy", split, qs, {"sample": fixtures_dir / "sample.md"}, tmp_path / "data"
        )
    config = {
        "dataset": {
            "name": "toy",
            "split": "test",
            "frozen": True,
            "data_dir": str(tmp_path / "data"),
        },
        "index": {
            "summarizer_model": "haiku",
            "allow_llm_inferred_structure": False,
            "path": str(tmp_path / "index"),
            "cache_dir": str(tmp_path / "cache"),
        },
        "answering": {"model": "sonnet", "max_context_chars": 2000, "max_nodes": 3},
        "jev": {"model": "jev-latest", "rate_per_million": 0.042, "max_spend_usd": 1.0},
        "llm": {
            "max_spend_usd": 5.0,
            "pricing": {
                "haiku": {"input_per_million": 1.0, "output_per_million": 5.0},
                "sonnet": {"input_per_million": 2.0, "output_per_million": 10.0},
            },
        },
        "thresholds": {"source": str(tmp_path / "thresholds.json")},
        "arms": ["A", "B", "C", "D", "E"],
        "baselines": {
            "dense": {
                "embedding_model": "bag",
                "reranker_model": "kw",
                "chunk_chars": 200,
                "chunk_overlap": 20,
                "embedding_rate_per_million": 0.1,
            },
            "llm_traversal": {"model": "sonnet"},
        },
        "fallback": {"model": "sonnet", "max_escalations_per_query": 3},
        "stats": {"bootstrap_resamples": 200, "seed": 0},
        "results_dir": str(tmp_path / "results"),
    }
    config.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def make_runner(config_path: Path) -> Runner:
    rates = {"haiku": LlmRate(1.0, 5.0), "sonnet": LlmRate(2.0, 10.0)}
    return Runner(
        config_path,
        jev_client=ScriptedJevClient(jev_handler),
        llm_client=ScriptedLlmClient(llm_handler, model_id="scripted", rates=rates),
        embedding=BagOfWords(),
        reranker=KeywordReranker(),
    )


def fitted_thresholds(fitted_on: str = "toy/dev", **kw: Any) -> Thresholds:
    base = Thresholds(
        tau_expand=0.5,
        tau_stop=0.9,
        tau_llm=0.4,
        beam_width=2,
        jev_model_id="jev-1.13.0",
        fitted_on=fitted_on,
    )
    return Thresholds(**{**base.to_dict(), **kw})


async def test_fit_picks_cheapest_config_meeting_recall(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    config = make_workspace(tmp_path, fixtures_dir)
    runner = make_runner(config)
    dev = load_split("toy", "dev", tmp_path / "data")
    traces, trees = await runner.collect_traces(
        dev, exploration_thresholds(Thresholds(beam_width=2))
    )
    assert set(traces) == {"d1", "d2"}
    result = await fit(
        dev,
        traces,
        trees,
        target_recall=0.95,
        jev_model_id="jev-1.13.0",
        base=Thresholds(beam_width=2),
        max_nodes=3,
        expand_grid=[0.3, 0.5, 0.7, 0.92],
        stop_grid=[0.5, 0.9],
    )
    assert result.target_met
    assert result.chosen.mean_recall == 1.0
    assert result.chosen.replay_misses == 0
    # 0.92 would drop Financial Statements (0.9) and lose recall; 0.5 and 0.7 tie on cost
    # and recall, and the tie-break prefers the more permissive floor.
    assert result.thresholds.tau_expand == pytest.approx(0.5)
    assert result.thresholds.fitted_on == "toy/dev"
    assert result.thresholds.jev_model_id == "jev-1.13.0"
    assert result.thresholds.prompt_version == PROMPT_VERSION
    assert 0.0 <= result.thresholds.tau_llm <= 1.0
    worst = max(result.grid, key=lambda p: p.tau_expand)
    assert worst.mean_recall < 1.0
    held = await verify(result.thresholds, dev, traces, trees, max_nodes=3)
    assert held["mean_recall"] == 1.0
    # Replay used no Jev calls beyond exploration.
    assert runner.jev.usage.requests == sum(len(t.expansions) for t in traces.values())


async def test_fit_refuses_unfrozen_split(tmp_path: Path, fixtures_dir: Path) -> None:
    make_workspace(tmp_path, fixtures_dir)
    dev = load_split("toy", "dev", tmp_path / "data")
    dev.frozen = False
    with pytest.raises(ValueError, match="not frozen"):
        await fit(dev, {}, {}, jev_model_id="x")


async def test_runner_end_to_end_writes_manifest(tmp_path: Path, fixtures_dir: Path) -> None:
    config = make_workspace(tmp_path, fixtures_dir)
    fitted_thresholds().save(tmp_path / "thresholds.json")
    runner = make_runner(config)
    run_dir = await runner.run()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["dataset"]["frozen"] and not manifest["exploratory"]
    assert manifest["threshold_fitted_on"] == "toy/dev"
    assert manifest["jev"]["resolved_model_ids"] == ["jev-1.13.0"]
    assert manifest["prompt_versions"]["questions"] == PROMPT_VERSION
    assert manifest["index"]["llm_inferred_structure_count_in_run"] == 0
    assert manifest["index"]["indexing_cost_usd_for_run_documents"] > 0
    summary = manifest["summary"]
    for arm in "ABCDE":
        s = summary[arm]
        assert s["queries"] == 3 and s["errors"] == 0, arm
        assert s["cost_usd"]["total_per_query"] > 0
    # Jev arms find both the capex and the segments sections.
    assert summary["D"]["section_recall"] == pytest.approx(2 / 3)
    assert summary["D"]["exact_match"] == pytest.approx(2 / 3)
    assert summary["C"]["section_recall"] == pytest.approx(2 / 3)
    assert summary["A"]["section_recall"] >= 2 / 3
    assert summary["B"]["cost_usd"]["arm_setup_total"] > 0
    assert summary["D"]["jev_model_ids"] == ["jev-1.13.0"]
    assert summary["E"]["escalation_rate"] >= 0.0
    assert any(c["a"] == "D" and c["b"] == "C" for c in manifest["comparisons"])
    assert manifest["calibration"]["decisions"] > 0
    for arm in "ABCDE":
        assert (run_dir / "raw" / f"outcomes_{arm}.jsonl").exists()
    traces = [
        json.loads(ln) for ln in (run_dir / "raw" / "traces_D.jsonl").read_text().splitlines()
    ]
    assert traces[0]["expansions"][0]["model_id"] == "jev-1.13.0"
    assert (run_dir / "summary.csv").read_text().count("\n") == 6
    text = render(run_dir)
    assert "jev_traversal" in text and "D vs C" in text
    written = write_calibration(run_dir, manifest)
    assert (run_dir / "calibration.csv") in written


async def test_limit_marks_run_exploratory(tmp_path: Path, fixtures_dir: Path) -> None:
    config = make_workspace(tmp_path, fixtures_dir)
    fitted_thresholds().save(tmp_path / "thresholds.json")
    run_dir = await make_runner(config).run(arms=["A"], limit=1)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["exploratory"] and manifest["dataset"]["query_count_run"] == 1


async def test_refuses_unfrozen_split(tmp_path: Path, fixtures_dir: Path) -> None:
    config = make_workspace(tmp_path, fixtures_dir)
    fitted_thresholds().save(tmp_path / "thresholds.json")
    doc = tmp_path / "data" / "toy" / "test" / "documents" / "sample.md"
    doc.write_text(doc.read_text() + "\nchanged")
    with pytest.raises(RunRefused, match="not frozen"):
        await make_runner(config).run(arms=["A"])


async def test_refuses_thresholds_fitted_on_evaluated_split(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    config = make_workspace(tmp_path, fixtures_dir)
    fitted_thresholds(fitted_on="toy/test").save(tmp_path / "thresholds.json")
    with pytest.raises(RunRefused, match="split under evaluation"):
        await make_runner(config).run(arms=["D"])


async def test_refuses_missing_model_id_and_unfitted_defaults(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    config = make_workspace(tmp_path, fixtures_dir)
    fitted_thresholds(jev_model_id=None).save(tmp_path / "thresholds.json")
    with pytest.raises(RunRefused, match="model ID"):
        await make_runner(config).run(arms=["D"])
    Thresholds().save(tmp_path / "thresholds.json")
    with pytest.raises(RunRefused, match="fitted_on"):
        await make_runner(config).run(arms=["C"])


async def test_refuses_thresholds_from_another_dataset(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    config = make_workspace(tmp_path, fixtures_dir)
    fitted_thresholds(fitted_on="other/dev").save(tmp_path / "thresholds.json")
    with pytest.raises(RunRefused, match="across datasets"):
        await make_runner(config).run(arms=["D"])


async def test_model_version_drift_marks_run_exploratory(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    config = make_workspace(tmp_path, fixtures_dir)
    fitted_thresholds(jev_model_id="jev-1.0.0").save(tmp_path / "thresholds.json")
    run_dir = await make_runner(config).run(arms=["D"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["exploratory"]
    assert "jev-1.0.0" in manifest["exploratory_reasons"][0]


async def test_refuses_prompt_version_mismatch(tmp_path: Path, fixtures_dir: Path) -> None:
    config = make_workspace(tmp_path, fixtures_dir)
    fitted_thresholds(prompt_version="v0").save(tmp_path / "thresholds.json")
    with pytest.raises(RunRefused, match="prompt"):
        await make_runner(config).run(arms=["D"])


async def test_refuses_zero_spend_cap(tmp_path: Path, fixtures_dir: Path) -> None:
    config = make_workspace(tmp_path, fixtures_dir)
    fitted_thresholds().save(tmp_path / "thresholds.json")
    runner = make_runner(config)
    runner.jev.max_spend_usd = 0.0
    with pytest.raises(RunRefused, match="max_spend_usd"):
        await runner.run(arms=["D"])
    runner = make_runner(config)
    runner.llm.max_spend_usd = 0.0
    with pytest.raises(RunRefused, match="answering"):
        await runner.run(arms=["A"])


async def test_refuses_document_without_structure(tmp_path: Path, fixtures_dir: Path) -> None:
    flat = tmp_path / "flat.md"
    flat.write_text("no headings here\n")
    freeze_split(
        "flat", "test", [EvalQuery("q", "flat", "?", "x")], {"flat": flat}, tmp_path / "data"
    )
    config = make_workspace(
        tmp_path,
        fixtures_dir,
        dataset={"name": "flat", "split": "test", "data_dir": str(tmp_path / "data")},
    )
    fitted_thresholds().save(tmp_path / "thresholds.json")
    with pytest.raises(RunRefused, match="native structure"):
        await make_runner(config).run(arms=["A"])


def test_write_manifest_requires_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        write_manifest(tmp_path, {"summary": {}})
