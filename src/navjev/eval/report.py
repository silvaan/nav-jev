"""Tables and the calibration figure from a run directory. Reads the manifest only."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{run_dir} has no manifest.json; nothing here is reportable")
    data: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return data


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summary_table(manifest: dict[str, Any]) -> str:
    rows = [s for s in manifest["summary"].values() if s.get("queries")]
    headers = [
        "arm",
        "name",
        "n",
        "recall",
        "EM",
        "F1",
        "lat p50 ms",
        "lat p95 ms",
        "exp",
        "esc",
        "$/query",
        "$/correct",
    ]
    lines = [" | ".join(headers)]
    for s in rows:
        cost = s["cost_usd"]
        lines.append(
            " | ".join(
                [
                    s["arm"],
                    s["name"],
                    str(s["queries"]),
                    _fmt(s["section_recall"]),
                    _fmt(s["exact_match"]),
                    _fmt(s["f1"]),
                    _fmt(s["latency_ms_median"], 0),
                    _fmt(s["latency_ms_p95"], 0),
                    _fmt(s["mean_expansions"], 1),
                    _fmt(s["escalation_rate"]),
                    _fmt(cost["total_per_query"], 5),
                    _fmt(cost["per_correct_answer"], 4),
                ]
            )
        )
    return "\n".join(lines)


def comparison_table(manifest: dict[str, Any]) -> str:
    lines = ["a vs b | metric | mean diff | 95% CI | n | distinguishable"]
    for c in manifest.get("comparisons", []):
        for metric, m in c["metrics"].items():
            lines.append(
                f"{c['a']} vs {c['b']} | {metric} | {_fmt(m['mean_diff'], 4)} | "
                f"[{_fmt(m['ci95'][0], 4)}, {_fmt(m['ci95'][1], 4)}] | {m['n']} | "
                f"{'yes' if m['distinguishable'] else 'no'}"
            )
    return "\n".join(lines)


def write_calibration(run_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    """Always writes calibration.csv; writes calibration.png when matplotlib is present."""
    cal = manifest.get("calibration", {})
    written: list[Path] = []
    if not cal.get("decisions"):
        return written
    csv_path = Path(run_dir) / "calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["curve", "predicted", "observed", "n"])
        for name in ("curve_probability", "curve_normalized_score"):
            for point in cal.get(name, []):
                writer.writerow([name, point["predicted"], point["observed"], point["n"]])
    written.append(csv_path)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return written
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    for name, label in (
        ("curve_probability", "P(top two levels)"),
        ("curve_normalized_score", "normalized score"),
    ):
        pts = cal.get(name, [])
        if pts:
            ax.plot(
                [p["predicted"] for p in pts],
                [p["observed"] for p in pts],
                marker="o",
                label=f"{label} (ECE {cal.get('ece_' + name.removeprefix('curve_'), 0):.3f})",
            )
    ax.set_xlabel("predicted")
    ax.set_ylabel("observed fraction with gold in subtree")
    ax.set_title(f"Jev child decisions, n={cal['decisions']}")
    ax.legend()
    png = Path(run_dir) / "calibration.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(png)
    return written


def render(run_dir: Path) -> str:
    manifest = load_manifest(run_dir)
    ds = manifest["dataset"]
    head = [
        f"run: {run_dir}",
        (
            f"dataset: {ds['name']}/{ds['split']}  hash={ds['content_hash'][:12]}  "
            f"queries={ds['query_count_run']}/{ds['query_count_in_split']}  "
            f"docs={ds['document_count']}"
        ),
        (
            f"exploratory: {manifest['exploratory']}"
            + (
                f" ({'; '.join(manifest.get('exploratory_reasons', []))})"
                if manifest["exploratory"]
                else ""
            )
            + f"  commit: {manifest['commit']}"
        ),
        (
            f"jev: {manifest['jev']['resolved_model_ids']}  thresholds fitted on: "
            f"{manifest['threshold_fitted_on']}"
        ),
        (
            "llm-inferred structure in run: "
            f"{manifest['index'].get('llm_inferred_structure_count_in_run')}"
        ),
        "",
        summary_table(manifest),
        "",
        comparison_table(manifest),
    ]
    cal = manifest.get("calibration", {})
    if cal.get("decisions"):
        head += [
            "",
            (
                f"calibration: {cal['decisions']} Jev child decisions, "
                f"ECE(probability)={cal['ece_probability']:.3f}, "
                f"ECE(normalized score)={cal['ece_normalized_score']:.3f}"
            ),
        ]
    return "\n".join(head)
