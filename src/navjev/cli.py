"""Typer CLI: build, ask, bench, fit, report, dataset."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
dataset_app = typer.Typer(help="Freeze and inspect evaluation splits.")
app.add_typer(dataset_app, name="dataset")


def _load_env() -> None:
    """Read `.env` into the environment without a dependency; existing values win."""
    import os

    env = Path(".env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _llm_client(config: str | None, max_spend_usd: float) -> object:
    from navjev.eval.config import RunConfig
    from navjev.llm import LlmClient

    rates = RunConfig.load(Path(config)).llm.rates() if config else {}
    if not rates:
        typer.echo(
            "note: no llm pricing loaded; LLM cost will be recorded as unknown", err=True
        )
    return LlmClient(rates=rates, max_spend_usd=max_spend_usd)


@app.command()
def build(
    doc: str,
    out: str = typer.Option(..., help="Index directory."),
    summarizer: str = "claude-haiku-4-5-20251001",
    config: str | None = typer.Option(None, help="Config to read LLM pricing from."),
    max_spend_usd: float = typer.Option(0.0, help="LLM spend cap; zero blocks every call."),
    allow_llm_structure: bool = typer.Option(
        False, help="Infer structure with an LLM when the file has none (tagged llm_inferred)."
    ),
) -> None:
    """Parse a document into a tree and summarize its nodes."""
    _load_env()
    from navjev.build.index import Index
    from navjev.build.parsers import LlmStructureParser, NoStructureError
    from navjev.llm import LlmClient

    llm = _llm_client(config, max_spend_usd)
    assert isinstance(llm, LlmClient)
    fallback = LlmStructureParser(llm, summarizer) if allow_llm_structure else None
    try:
        index = Index.build(Path(doc), Path(out), summarizer, llm, structure_fallback=fallback)
    except NoStructureError as error:
        typer.echo(f"refused: {error}", err=True)
        raise typer.Exit(2) from None
    doc_id = Path(doc).stem
    tree = index.load_tree(doc_id)
    entry = index.manifest()["documents"][doc_id]  # type: ignore[index]
    if tree.structure_source == "llm_inferred":
        typer.echo(
            "WARNING: structure was inferred by an LLM; the index is tagged llm_inferred"
        )
    typer.echo(
        f"indexed {doc_id}: {len(tree.nodes)} nodes, depth {tree.max_depth()}, "
        f"parser={tree.parser}, summarization={json.dumps(entry['summarization'])}"
    )
    typer.echo(f"tree: {index.docs_dir / (doc_id + '.json')}")


@app.command()
def ask(
    index: str,
    query: str,
    doc: str | None = typer.Option(None, help="Document id; defaults to the only one."),
    thresholds: str | None = typer.Option(
        None, help="Fitted thresholds JSON; defaults otherwise."
    ),
    max_spend_usd: float = typer.Option(0.0, help="Jev spend cap; zero blocks every call."),
    jev_model: str = "jev-latest",
    rate_per_million: float = typer.Option(0.042, help="Jev price, for cost accounting."),
    max_nodes: int = 5,
    show_trace: bool = True,
    replay: str | None = typer.Option(
        None, help="Replay a recorded fixture (JSONL) instead of calling the API."
    ),
    record: str | None = typer.Option(None, help="Append every live response to this JSONL."),
) -> None:
    """Retrieve with the Jev policy and print the path it walked."""
    _load_env()
    from navjev.build.index import Index
    from navjev.jev import JevClient, RecordedJevClient
    from navjev.traverse.beam import BeamSearch, JevPolicy
    from navjev.traverse.questions import Thresholds

    idx = Index(Path(index))
    ids = idx.doc_ids()
    if doc is None:
        if len(ids) != 1:
            typer.echo(f"pass --doc; the index holds {ids}", err=True)
            raise typer.Exit(2)
        doc = ids[0]
    tree = idx.load_tree(doc)
    t = Thresholds.load(Path(thresholds)) if thresholds else Thresholds().without_fallback()
    client: JevClient
    if replay:
        client = RecordedJevClient(replay, model=jev_model, rate_per_million=rate_per_million)
    else:
        client = JevClient(
            model=jev_model,
            max_spend_usd=max_spend_usd,
            rate_per_million=rate_per_million,
            record_to=record,
        )
    search = BeamSearch(JevPolicy(client, t.without_fallback()), t.without_fallback())
    result = asyncio.run(search.retrieve(tree, query, max_nodes))
    if show_trace:
        typer.echo(result.trace.as_path_text(tree))
        typer.echo("")
    if result.no_answer:
        typer.echo("no answer: the document was judged off-topic for this query")
    for node, score in zip(result.nodes, result.scores, strict=True):
        pages = f" pages {node.page_span[0]}-{node.page_span[1]}" if node.page_span else ""
        typer.echo(f"[{score:.2f}] {' > '.join(tree.path_to(node.id))}{pages}")
    typer.echo(
        f"\n{len(result.trace.expansions)} expansions, {result.trace.total_input_tokens} input "
        f"tokens, ${result.cost_usd:.6f}, {result.latency_ms:.0f} ms, model={result.trace.model_ids}"
    )


@app.command()
def bench(
    config: str,
    arms: str = typer.Option("", help="Comma-separated subset of the config's arms."),
    limit: int = typer.Option(0, help="Truncate the split; marks the run exploratory."),
) -> None:
    """Run the arms in a config over a frozen split and write a manifest."""
    _load_env()
    from navjev.eval.report import render, write_calibration
    from navjev.eval.runner import Runner, RunRefused

    runner = Runner(Path(config))
    try:
        run_dir = asyncio.run(
            runner.run([a.strip() for a in arms.split(",") if a.strip()] or None, limit or None)
        )
    except RunRefused as error:
        typer.echo(f"refused: {error}", err=True)
        raise typer.Exit(2) from None
    from navjev.eval.report import load_manifest

    write_calibration(run_dir, load_manifest(run_dir))
    typer.echo(render(run_dir))


@app.command()
def fit(
    config: str,
    split: str = "dev",
    target_recall: float = 0.95,
    limit: int = typer.Option(0, help="Fit on the first N dev queries only."),
) -> None:
    """Fit thresholds on a dev split and write them where bench can load them."""
    _load_env()
    from navjev.eval.datasets import load_split
    from navjev.eval.runner import Runner, RunRefused
    from navjev.eval.thresholds import exploration_thresholds
    from navjev.eval.thresholds import fit as fit_thresholds
    from navjev.traverse.questions import Thresholds

    runner = Runner(Path(config))
    cfg = runner.config
    if split == cfg.dataset.split:
        typer.echo(f"refused: fitting on {split}, the split the config evaluates", err=True)
        raise typer.Exit(2)
    dev = load_split(cfg.dataset.name, split, Path(cfg.dataset.data_dir))
    if not dev.frozen:
        typer.echo("refused: the dev split is not frozen", err=True)
        raise typer.Exit(2)

    async def go() -> None:
        base = Thresholds()
        traces, trees = await runner.collect_traces(
            dev, exploration_thresholds(base), limit or None
        )
        model_ids = runner.jev.usage.model_ids
        if len(model_ids) != 1:
            typer.echo(
                f"refused: exploration saw model ids {model_ids}; thresholds need exactly one",
                err=True,
            )
            raise typer.Exit(2)
        result = await fit_thresholds(
            dev, traces, trees, target_recall, model_ids[0], base, cfg.answering.max_nodes
        )
        out = Path(cfg.thresholds.source)
        result.thresholds.save(out)
        out.with_suffix(".fit.json").write_text(json.dumps(result.to_dict(), indent=2))
        typer.echo(json.dumps(result.thresholds.to_dict(), indent=2))
        typer.echo(
            f"target recall {target_recall} {'met' if result.target_met else 'NOT met'}: "
            f"recall={result.chosen.mean_recall:.3f} expansions={result.chosen.mean_expansions:.2f} "
            f"replay_misses={result.chosen.replay_misses} on {result.chosen.queries} queries; "
            f"tau_llm escalates {result.escalation_rate_at_tau_llm:.1%} of dev expansions"
        )
        typer.echo(
            f"wrote {out} and {out.with_suffix('.fit.json')}; jev spent ${runner.jev.spent_usd:.4f}"
        )

    try:
        asyncio.run(go())
    except RunRefused as error:
        typer.echo(f"refused: {error}", err=True)
        raise typer.Exit(2) from None


@app.command()
def report(run_dir: str) -> None:
    """Render tables and the calibration figure from a run directory."""
    from navjev.eval.report import load_manifest, render, write_calibration

    path = Path(run_dir)
    written = write_calibration(path, load_manifest(path))
    typer.echo(render(path))
    for w in written:
        typer.echo(f"wrote {w}")


@dataset_app.command("freeze")
def dataset_freeze(
    name: str,
    data_dir: str = "data",
    cache_dir: str = ".cache/datasets",
    max_queries: int = typer.Option(0, help="Cap per split at freeze time (0 = all)."),
) -> None:
    """Download a dataset, write dev and test splits, and record their hashes."""
    from navjev.eval.datasets import LOADERS

    if name not in LOADERS:
        typer.echo(f"unknown dataset {name}; known: {sorted(LOADERS)}", err=True)
        raise typer.Exit(2)
    splits = LOADERS[name](Path(cache_dir), Path(data_dir), max_queries or None)
    for split_name, split in splits.items():
        typer.echo(
            f"{name}/{split_name}: {len(split.queries)} queries, {len(split.documents)} documents, "
            f"hash={split.content_hash[:12]}"
        )


@dataset_app.command("show")
def dataset_show(name: str, split: str = "test", data_dir: str = "data") -> None:
    """Verify a frozen split's hash and print its size."""
    from navjev.eval.datasets import load_split

    s = load_split(name, split, Path(data_dir))
    typer.echo(
        f"{name}/{split}: {len(s.queries)} queries, {len(s.documents)} documents, "
        f"hash={s.content_hash[:12]}, frozen={'yes' if s.frozen else 'NO (hash mismatch)'}"
    )


if __name__ == "__main__":
    app()
