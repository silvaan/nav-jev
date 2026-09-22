"""Two commands: `nav-jev add <file>` and `nav-jev ask "<question>"`."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from navjev.navigator import Navigator

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)

FILES_ARGUMENT = typer.Argument(..., help="Markdown, PDF or DOCX files to index.")
INDEX_OPTION = typer.Option("index", "--index", "-i", help="Index directory.")
SPEND_OPTION = typer.Option(None, "--max-spend-usd", help="Stop once this much was spent.")


@app.command()
def add(
    files: list[Path] = FILES_ARGUMENT,
    index: str = INDEX_OPTION,
    summarizer: str = typer.Option("gpt-4.1-mini", help="Model that writes section summaries."),
    allow_llm_structure: bool = typer.Option(
        False,
        help="Infer an outline with the LLM when the file has none (tagged llm_inferred).",
    ),
    replace: bool = typer.Option(False, help="Re-index documents already in the index."),
    max_spend_usd: float | None = SPEND_OPTION,
) -> None:
    """Parse documents into section trees and summarize every section."""
    from navjev.build.parsers import NoStructureError

    nav = Navigator(
        index,
        summarizer=summarizer,
        allow_llm_structure=allow_llm_structure,
        max_spend_usd=max_spend_usd,
    )
    for file in files:
        try:
            doc = nav.add(file, replace=replace)
        except NoStructureError as error:
            typer.echo(f"skipped: {error}", err=True)
            continue
        tree = nav.tree(doc)
        note = (
            "  (structure inferred by the LLM)"
            if tree.structure_source == "llm_inferred"
            else ""
        )
        typer.echo(f"{doc}: {len(tree.nodes)} sections, depth {tree.max_depth()}{note}")
    llm = nav.usage["llm"]
    cost = (
        f"{llm['input_tokens']} in / {llm['output_tokens']} out tokens"
        if llm["unpriced_models"]
        else f"${llm['spent_usd']:.4f}"
    )
    typer.echo(f"summaries: {llm['requests']} LLM calls, {cost}")


@app.command()
def ask(
    question: str,
    index: str = INDEX_OPTION,
    doc: str | None = typer.Option(None, help="Search one document instead of all."),
    top_k: int = typer.Option(5, help="How many sections to return."),
    detail: bool = typer.Option(False, help="Print every decision the walk made."),
    as_json: bool = typer.Option(False, "--json", help="Print the raw result as JSON."),
    max_spend_usd: float | None = SPEND_OPTION,
    replay: str | None = typer.Option(None, hidden=True, help="Replay a recorded Jev fixture."),
) -> None:
    """Find the sections most likely to answer the question."""
    kwargs = {}
    if replay:
        from navjev.jev import RecordedJevClient

        kwargs["jev_client"] = RecordedJevClient(replay)
    nav = Navigator(index, max_spend_usd=max_spend_usd, **kwargs)  # type: ignore[arg-type]
    result = nav.search(question, doc=doc, top_k=top_k, detail=detail)
    if as_json:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if detail:
        for trace in result["traces"]:
            typer.echo(trace["text"])
            typer.echo("")
    if result["no_answer"]:
        typer.echo("no answer: the question is not about anything in the index")
    for hit in result["results"]:
        pages = f"  pages {hit['pages'][0]}-{hit['pages'][1]}" if hit["pages"] else ""
        prefix = f"{hit['doc']}: " if len(nav.docs) > 1 else ""
        typer.echo(f"[{hit['score']:.2f}] {prefix}{' > '.join(hit['path'])}{pages}")
    typer.echo(f"cost ${result['cost_usd']:.5f}")


if __name__ == "__main__":
    app()
