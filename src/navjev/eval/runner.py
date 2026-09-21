"""Runs one config: every arm over one frozen split, writing JSONL, CSV and a manifest.

Refusals, all deliberate: an unfrozen split; thresholds fitted on the split being
evaluated; a missing Jev model ID; a spend cap of zero on an arm that needs paid calls;
a config whose prompt version does not match questions.py.
"""

from __future__ import annotations

import asyncio
import csv
import datetime as dt
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from navjev.baselines.bm25 import Bm25Retriever
from navjev.baselines.dense import (
    CrossEncoderReranker,
    DenseRetriever,
    EmbeddingBackend,
    Reranker,
    make_embedding_backend,
)
from navjev.baselines.llm_traversal import LlmPolicy
from navjev.build.index import Index
from navjev.build.parsers import LlmStructureParser, NoStructureError
from navjev.build.summarize import SUMMARY_PROMPT_VERSION, Summarizer, SummaryCache
from navjev.eval.answer import ANSWER_PROMPT_VERSION, answer_question, build_context
from navjev.eval.config import RunConfig
from navjev.eval.datasets import EvalQuery, Split, load_split, resolve_gold
from navjev.eval.metrics import (
    QueryOutcome,
    calibration_curve,
    exact_match,
    expected_calibration_error,
    paired_bootstrap,
    percentile,
    section_recall,
    token_f1,
)
from navjev.jev import JevBudgetExceeded, JevClient, JevError
from navjev.llm import LlmBudgetExceeded, LlmClient, LlmError
from navjev.traverse.beam import BeamSearch, JevPolicy, LlmFallback, TraversalBudgetExceeded
from navjev.traverse.questions import PROMPT_VERSION, Thresholds
from navjev.types import DocumentTree, RetrievalResult, Trace

ARM_NAMES = {
    "A": "bm25",
    "B": "dense+reranker",
    "C": "llm_traversal",
    "D": "jev_traversal",
    "E": "jev_traversal+fallback",
}
TREE_ARMS = ("C", "D", "E")
JEV_ARMS = ("D", "E")
COMPARISONS = (("D", "C"), ("E", "C"), ("D", "A"), ("D", "B"), ("E", "D"))


class RunRefused(RuntimeError):
    """The run would produce a number the protocol forbids reporting."""


@dataclass
class Retriever:
    arm: str
    retrieve: Any  # async callable(tree, query, max_nodes) -> RetrievalResult
    cost_per_query_token: float = 0.0
    fixed_cost_usd: float = 0.0
    """Arm-level setup cost (dense embedding of the corpus), amortized per query."""


class Runner:
    def __init__(
        self,
        config_path: Path,
        jev_client: JevClient | None = None,
        llm_client: LlmClient | None = None,
        embedding: EmbeddingBackend | None = None,
        reranker: Reranker | None = None,
        clock: Any = time.perf_counter,
    ) -> None:
        self.config = RunConfig.load(Path(config_path))
        cfg = self.config
        self.jev = jev_client or JevClient(
            model=cfg.jev.model,
            max_concurrency=cfg.jev.max_concurrency,
            max_spend_usd=cfg.jev.max_spend_usd,
            rate_per_million=cfg.jev.rate_per_million,
            timeout_s=cfg.jev.timeout_s,
        )
        self.llm = llm_client or LlmClient(
            rates=cfg.llm.rates(),
            max_spend_usd=cfg.llm.max_spend_usd,
            max_concurrency=cfg.llm.max_concurrency,
        )
        self._embedding = embedding
        self._reranker = reranker
        self._clock = clock

    # -- refusals ---------------------------------------------------------------------

    def check(self, split: Split, arms: Sequence[str]) -> Thresholds | None:
        cfg = self.config
        if cfg.dataset.frozen and not split.frozen:
            raise RunRefused(
                f"split {cfg.dataset.name}/{split.name} is not frozen or its hash changed"
            )
        needs_tree = any(a in TREE_ARMS for a in arms)
        thresholds: Thresholds | None = None
        if needs_tree:
            source = Path(cfg.thresholds.source)
            if not source.exists():
                raise RunRefused(f"thresholds file {source} does not exist; run `nav-jev fit`")
            thresholds = Thresholds.load(source)
            if thresholds.fitted_on is None:
                raise RunRefused(
                    "thresholds carry no `fitted_on`; unfitted defaults are not a result"
                )
            fitted_dataset, _, fitted_split = thresholds.fitted_on.rpartition("/")
            if fitted_split == split.name:
                raise RunRefused(
                    f"thresholds were fitted on {thresholds.fitted_on}, the split under evaluation"
                )
            if fitted_dataset and fitted_dataset != cfg.dataset.name:
                raise RunRefused(
                    f"thresholds were fitted on {thresholds.fitted_on}, not on "
                    f"{cfg.dataset.name}; thresholds do not transfer across datasets, refit"
                )
            if thresholds.prompt_version != PROMPT_VERSION:
                raise RunRefused(
                    f"thresholds were fitted for prompt {thresholds.prompt_version}, "
                    f"questions.py is {PROMPT_VERSION}; refit"
                )
        if any(a in JEV_ARMS for a in arms):
            assert thresholds is not None
            if not thresholds.jev_model_id:
                raise RunRefused("thresholds carry no Jev model ID; refit and record it")
            if self.jev.max_spend_usd <= 0.0:
                raise RunRefused("jev.max_spend_usd is zero; arms D/E need paid calls")
        if self.llm.max_spend_usd <= 0.0:
            raise RunRefused("llm.max_spend_usd is zero; every arm needs the answering model")
        if "E" in arms and cfg.fallback is None:
            raise RunRefused("arm E needs a `fallback` section in the config")
        if "E" in arms and thresholds is not None and not thresholds.fallback_enabled:
            raise RunRefused("arm E needs tau_llm > 0 in the fitted thresholds")
        if "C" in arms and cfg.baselines.llm_traversal is None:
            raise RunRefused("arm C needs `baselines.llm_traversal`")
        if "B" in arms and cfg.baselines.dense is None:
            raise RunRefused("arm B needs `baselines.dense`")
        return thresholds

    # -- index ------------------------------------------------------------------------

    async def ensure_index(self, split: Split) -> Index:
        cfg = self.config
        index = Index(cfg.index_path())
        cache = SummaryCache(Path(cfg.index.cache_dir))
        summarizer = Summarizer(cfg.index.summarizer_model, cache, self.llm)
        fallback = (
            LlmStructureParser(self.llm, cfg.index.summarizer_model)
            if cfg.index.allow_llm_inferred_structure
            else None
        )
        missing: list[str] = []
        for doc_id in split.doc_ids:
            if index.has(doc_id):
                continue
            try:
                await index.add_document(
                    split.documents[doc_id], summarizer, fallback, doc_id=doc_id
                )
            except NoStructureError as error:
                missing.append(f"{doc_id}: {error}")
        if missing:
            raise RunRefused(
                "documents without native structure (set index.allow_llm_inferred_structure "
                "for an explicitly labelled run):\n  " + "\n  ".join(missing)
            )
        return index

    # -- arms -------------------------------------------------------------------------

    def _retriever(
        self, arm: str, trees: dict[str, DocumentTree], thresholds: Thresholds | None
    ) -> Retriever:
        cfg = self.config
        if arm == "A":
            by_doc = {d: Bm25Retriever(t) for d, t in trees.items()}

            async def bm25(tree: DocumentTree, query: str, max_nodes: int) -> RetrievalResult:
                return await asyncio.to_thread(by_doc[tree.doc_id].retrieve, query, max_nodes)

            return Retriever(arm, bm25)

        if arm == "B":
            dense_cfg = cfg.baselines.dense
            assert dense_cfg is not None
            embedding = self._embedding or make_embedding_backend(dense_cfg.embedding_model)
            reranker = self._reranker or (
                CrossEncoderReranker(dense_cfg.reranker_model)
                if dense_cfg.reranker_model
                else None
            )
            dense = {
                d: DenseRetriever(
                    t,
                    embedding,
                    reranker,
                    dense_cfg.chunk_chars,
                    dense_cfg.chunk_overlap,
                    dense_cfg.rerank_candidates,
                )
                for d, t in trees.items()
            }
            rate = dense_cfg.embedding_rate_per_million / 1_000_000
            fixed = sum(r.index_embedding_tokens for r in dense.values()) * rate

            async def dense_retrieve(
                tree: DocumentTree, query: str, max_nodes: int
            ) -> RetrievalResult:
                return await asyncio.to_thread(dense[tree.doc_id].retrieve, query, max_nodes)

            return Retriever(
                arm, dense_retrieve, cost_per_query_token=rate, fixed_cost_usd=fixed
            )

        assert thresholds is not None
        if arm == "C":
            assert cfg.baselines.llm_traversal is not None
            policy: Any = LlmPolicy(
                self.llm, cfg.baselines.llm_traversal.model, thresholds.without_fallback()
            )
            search = BeamSearch(policy, thresholds.without_fallback())
        elif arm == "D":
            policy = JevPolicy(self.jev, thresholds.without_fallback())
            search = BeamSearch(policy, thresholds.without_fallback())
        elif arm == "E":
            assert cfg.fallback is not None
            fallback = LlmFallback(
                self.llm, cfg.fallback.model, cfg.fallback.max_escalations_per_query
            )
            policy = JevPolicy(self.jev, thresholds, fallback)
            search = BeamSearch(policy, thresholds)
        else:
            raise ValueError(arm)

        async def tree_retrieve(
            tree: DocumentTree, query: str, max_nodes: int
        ) -> RetrievalResult:
            return await search.retrieve(tree, query, max_nodes)

        return Retriever(arm, tree_retrieve)

    # -- one query --------------------------------------------------------------------

    async def _run_query(
        self, retriever: Retriever, query: EvalQuery, tree: DocumentTree
    ) -> tuple[QueryOutcome, Trace | None]:
        cfg = self.config
        gold = resolve_gold(query, tree)
        outcome = QueryOutcome(
            query_id=query.query_id,
            arm=retriever.arm,
            retrieved_node_ids=[],
            answer="",
            latency_ms=0.0,
            cost_usd=0.0,
            input_tokens=0,
            escalated=False,
            jev_model_id=None,
            gold_node_ids=gold,
        )
        trace: Trace | None = None
        try:
            started = self._clock()
            result = await retriever.retrieve(tree, query.question, cfg.answering.max_nodes)
            outcome.latency_ms = (self._clock() - started) * 1000.0
            trace = result.trace
            outcome.retrieved_node_ids = result.node_ids
            outcome.cost_usd = (
                result.cost_usd + result.query_tokens * retriever.cost_per_query_token
            )
            outcome.input_tokens = result.trace.total_input_tokens + result.query_tokens
            outcome.expansions = len(result.trace.expansions)
            outcome.escalations = sum(1 for e in result.trace.expansions if e.escalated_to_llm)
            outcome.escalated = outcome.escalations > 0
            outcome.no_answer = result.no_answer
            jev_ids = [
                e.model_id.split("+")[0]
                for e in result.trace.expansions
                if retriever.arm in JEV_ARMS
            ]
            outcome.jev_model_id = ",".join(sorted(set(jev_ids))) if jev_ids else None
            nodes = result.nodes
        except (TraversalBudgetExceeded, JevError, LlmError, ValueError) as error:
            if isinstance(error, (JevBudgetExceeded, LlmBudgetExceeded)):
                raise
            outcome.error = f"{type(error).__name__}: {error}"
            nodes = []

        context = build_context(
            nodes, [tree.path_to(n.id) for n in nodes], cfg.answering.max_context_chars
        )
        try:
            answered = await answer_question(
                self.llm, cfg.answering.model, query.question, context
            )
            outcome.answer = answered.answer
            outcome.answer_cost_usd = answered.cost_usd
            outcome.answer_latency_ms = answered.latency_ms
            outcome.answer_input_tokens = answered.input_tokens
            outcome.answer_output_tokens = answered.output_tokens
        except LlmBudgetExceeded:
            raise
        except LlmError as error:
            outcome.error = (outcome.error + "; " if outcome.error else "") + f"answer: {error}"

        if gold:
            outcome.section_recall = section_recall(outcome.retrieved_node_ids, gold)
        golds = query.answer if isinstance(query.answer, list) else [query.answer]
        if golds:
            outcome.exact_match = exact_match(outcome.answer, golds)
            outcome.f1 = token_f1(outcome.answer, golds)
        return outcome, trace

    # -- exploration, for threshold fitting ------------------------------------------

    async def collect_traces(
        self, split: Split, thresholds: Thresholds, limit: int | None = None
    ) -> tuple[dict[str, Trace], dict[str, DocumentTree]]:
        """Traverse a split with the Jev policy at the given (exploration) thresholds and
        return one trace per query. Used by `fit` and `verify`; no answering, no
        manifest, and nothing here is a reportable number."""
        if self.jev.max_spend_usd <= 0.0:
            raise RunRefused("jev.max_spend_usd is zero; exploration needs paid calls")
        if self.llm.max_spend_usd <= 0.0 and not all(
            Index(self.config.index_path()).has(d) for d in split.doc_ids
        ):
            raise RunRefused("llm.max_spend_usd is zero and the index needs summaries")
        index = await self.ensure_index(split)
        queries = list(split.queries)[: limit or None]
        trees = {d: index.load_tree(d) for d in sorted({q.doc_id for q in queries})}
        search = BeamSearch(
            JevPolicy(self.jev, thresholds.without_fallback()), thresholds.without_fallback()
        )
        traces: dict[str, Trace] = {}
        for q in queries:
            try:
                result = await search.retrieve(
                    trees[q.doc_id], q.question, self.config.answering.max_nodes
                )
            except TraversalBudgetExceeded:
                continue
            traces[q.query_id] = result.trace
        return traces, trees

    # -- the run ----------------------------------------------------------------------

    async def run(self, arms: list[str] | None = None, limit: int | None = None) -> Path:
        """Returns the path to the run directory. Arms default to every arm in the
        config. `limit` marks the run exploratory in the manifest, since a truncated
        split is not the frozen split."""
        cfg = self.config
        arms = list(arms or cfg.arms)
        split = load_split(cfg.dataset.name, cfg.dataset.split, Path(cfg.dataset.data_dir))
        thresholds = self.check(split, arms)
        queries = list(split.queries)
        exploratory_reasons: list[str] = []
        if limit is not None and limit < len(queries):
            queries = queries[:limit]
            exploratory_reasons.append(f"limit={limit} truncates the frozen split")

        index = await self.ensure_index(split)
        doc_ids = sorted({q.doc_id for q in queries})
        trees = {d: index.load_tree(d) for d in doc_ids}

        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(cfg.results_dir) / f"{cfg.dataset.name}-{split.name}-{stamp}"
        (run_dir / "raw").mkdir(parents=True, exist_ok=True)

        outcomes: dict[str, list[QueryOutcome]] = {}
        traces: dict[str, dict[str, Trace]] = {}
        arm_setup: dict[str, dict[str, Any]] = {}
        for arm in arms:
            retriever = self._retriever(arm, trees, thresholds)
            arm_setup[arm] = {"fixed_cost_usd": retriever.fixed_cost_usd}
            arm_outcomes: list[QueryOutcome] = []
            arm_traces: dict[str, Trace] = {}
            for q in queries:
                outcome, trace = await self._run_query(retriever, q, trees[q.doc_id])
                arm_outcomes.append(outcome)
                if trace is not None:
                    arm_traces[q.query_id] = trace
            outcomes[arm] = arm_outcomes
            traces[arm] = arm_traces
            with (run_dir / "raw" / f"outcomes_{arm}.jsonl").open("w", encoding="utf-8") as fh:
                for o in arm_outcomes:
                    fh.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")
            with (run_dir / "raw" / f"traces_{arm}.jsonl").open("w", encoding="utf-8") as fh:
                for qid, tr in arm_traces.items():
                    fh.write(
                        json.dumps({"query_id": qid, **tr.to_dict()}, ensure_ascii=False) + "\n"
                    )

        # Thresholds are only valid against the model version they were fitted on. A
        # different version answering makes the run exploratory, not reportable.
        if thresholds is not None and any(a in JEV_ARMS for a in arms):
            seen = set(self.jev.usage.model_ids)
            if seen and seen != {thresholds.jev_model_id}:
                exploratory_reasons.append(
                    f"thresholds fitted on {thresholds.jev_model_id}, run answered by {sorted(seen)}"
                )
        exploratory = bool(exploratory_reasons)

        index_manifest = index.manifest()
        index_docs: dict[str, dict[str, Any]] = dict(index_manifest["documents"])  # type: ignore[call-overload]
        indexing_cost = 0.0
        for doc_id in doc_ids:
            entry = index_docs[doc_id]
            indexing_cost += float((entry.get("summarization") or {}).get("cost_usd", 0.0))
            indexing_cost += float(entry.get("structure_cost_usd", 0.0))
        summaries = {
            arm: arm_summary(
                arm, outcomes[arm], indexing_cost, arm_setup[arm]["fixed_cost_usd"]
            )
            for arm in arms
        }
        comparisons = pairwise(outcomes, cfg.stats.bootstrap_resamples, cfg.stats.seed)
        calibration = calibration_from_traces(traces, trees, queries)
        write_summary_csv(run_dir / "summary.csv", summaries)

        payload: dict[str, Any] = {
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "hostname": socket.gethostname(),
            "commit": git_commit(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
            "config_path": cfg.path,
            "config": cfg.raw_text,
            "dataset": {
                "name": cfg.dataset.name,
                "split": split.name,
                "content_hash": split.content_hash,
                "frozen": split.frozen,
                "query_count_in_split": len(split.queries),
                "query_count_run": len(queries),
                "document_count": len(doc_ids),
                "provenance": split.provenance,
            },
            "exploratory": exploratory,
            "exploratory_reasons": exploratory_reasons,
            "limit": limit,
            "arms": {a: ARM_NAMES[a] for a in arms},
            "prompt_versions": {
                "questions": PROMPT_VERSION,
                "summary": SUMMARY_PROMPT_VERSION,
                "answer": ANSWER_PROMPT_VERSION,
            },
            "thresholds": thresholds.to_dict() if thresholds else None,
            "threshold_source": cfg.thresholds.source if thresholds else None,
            "threshold_fitted_on": thresholds.fitted_on if thresholds else None,
            "index": {k: v for k, v in index_manifest.items() if k != "documents"}
            | {
                "documents_in_run": {d: index_docs[d] for d in doc_ids},
                "indexing_cost_usd_for_run_documents": indexing_cost,
                "llm_inferred_structure_count_in_run": sum(
                    1 for d in doc_ids if trees[d].structure_source == "llm_inferred"
                ),
            },
            "jev": {
                "requested_model": cfg.jev.model,
                "resolved_model_ids": self.jev.usage.model_ids,
                "rate_per_million": cfg.jev.rate_per_million,
                "requests": self.jev.usage.requests,
                "input_tokens": self.jev.usage.input_tokens,
                "spent_usd": self.jev.spent_usd,
                "max_spend_usd": self.jev.max_spend_usd,
            },
            "llm": {
                "requests": self.llm.usage.requests,
                "input_tokens": self.llm.usage.input_tokens,
                "output_tokens": self.llm.usage.output_tokens,
                "spent_usd": self.llm.spent_usd,
                "max_spend_usd": self.llm.max_spend_usd,
                "unpriced_models": self.llm.usage.unpriced_models,
                "pricing": {m: r.__dict__ for m, r in cfg.llm.rates().items()},
            },
            "summary": summaries,
            "comparisons": comparisons,
            "calibration": calibration,
        }
        write_manifest(run_dir, payload)
        return run_dir


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def arm_summary(
    arm: str, outcomes: Sequence[QueryOutcome], indexing_cost_usd: float, fixed_cost_usd: float
) -> dict[str, Any]:
    n = len(outcomes)
    if n == 0:
        return {"arm": arm, "name": ARM_NAMES[arm], "queries": 0}
    retrieval_cost = sum(o.cost_usd for o in outcomes)
    answer_cost = sum(o.answer_cost_usd for o in outcomes)
    amortized_index = (indexing_cost_usd + fixed_cost_usd) / n
    total_cost = retrieval_cost + answer_cost + indexing_cost_usd + fixed_cost_usd
    correct = sum(1 for o in outcomes if o.exact_match == 1.0)
    latencies = [o.latency_ms for o in outcomes]
    with_recall = [o for o in outcomes if o.section_recall is not None]
    return {
        "arm": arm,
        "name": ARM_NAMES[arm],
        "queries": n,
        "errors": sum(1 for o in outcomes if o.error),
        "no_answer": sum(1 for o in outcomes if o.no_answer),
        "section_recall": _mean([o.section_recall for o in outcomes]),
        "section_recall_queries": len(with_recall),
        "mean_retrieved_nodes": sum(len(o.retrieved_node_ids) for o in outcomes) / n,
        "exact_match": _mean([o.exact_match for o in outcomes]),
        "f1": _mean([o.f1 for o in outcomes]),
        "latency_ms_median": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "answer_latency_ms_median": percentile([o.answer_latency_ms for o in outcomes], 50),
        "mean_expansions": sum(o.expansions for o in outcomes) / n,
        "escalation_rate": (
            sum(o.escalations for o in outcomes) / max(1, sum(o.expansions for o in outcomes))
        ),
        "queries_with_escalation": sum(1 for o in outcomes if o.escalated),
        "retrieval_input_tokens_per_query": sum(o.input_tokens for o in outcomes) / n,
        "cost_usd": {
            "retrieval_per_query": retrieval_cost / n,
            "answering_per_query": answer_cost / n,
            "indexing_amortized_per_query": amortized_index,
            "indexing_total": indexing_cost_usd,
            "arm_setup_total": fixed_cost_usd,
            "total_per_query": total_cost / n,
            "per_correct_answer": (total_cost / correct) if correct else None,
            "correct_answers": correct,
        },
        "jev_model_ids": sorted({o.jev_model_id for o in outcomes if o.jev_model_id}),
    }


def pairwise(
    outcomes: dict[str, list[QueryOutcome]], n_resamples: int, seed: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a, b in COMPARISONS:
        if a not in outcomes or b not in outcomes:
            continue
        pa = {o.query_id: o for o in outcomes[a]}
        pb = {o.query_id: o for o in outcomes[b]}
        shared = [q for q in pa if q in pb]
        if not shared:
            continue
        entry: dict[str, Any] = {"a": a, "b": b, "queries": len(shared), "metrics": {}}
        getters: list[tuple[str, Callable[[QueryOutcome], float | None]]] = [
            ("section_recall", lambda o: o.section_recall),
            ("exact_match", lambda o: o.exact_match),
            ("f1", lambda o: o.f1),
            ("latency_ms", lambda o: o.latency_ms),
            ("total_cost_usd", lambda o: o.total_cost_usd),
        ]
        for metric, getter in getters:
            xs: list[float] = []
            ys: list[float] = []
            for q in shared:
                x, y = getter(pa[q]), getter(pb[q])
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            if not xs:
                continue
            diff, (lo, hi) = paired_bootstrap(xs, ys, n_resamples, seed)
            entry["metrics"][metric] = {
                "mean_diff": diff,
                "ci95": [lo, hi],
                "n": len(xs),
                "distinguishable": bool(lo > 0 or hi < 0),
            }
        out.append(entry)
    return out


def calibration_from_traces(
    traces: dict[str, dict[str, Trace]],
    trees: dict[str, DocumentTree],
    queries: Sequence[EvalQuery],
) -> dict[str, Any]:
    """Every Jev child decision from arms D and E, bucketed by the probability Jev put on
    the two top relevance levels ("holds part or all of the evidence"), against whether
    the child's subtree held a gold section. Escalated expansions contribute the Jev
    answers the LLM overrode, since the question is about Jev's calibration."""
    by_query = {q.query_id: q for q in queries}
    predicted: list[float] = []
    outcomes: list[bool] = []
    normalized: list[float] = []
    for arm in JEV_ARMS:
        for qid, trace in traces.get(arm, {}).items():
            q = by_query[qid]
            tree = trees[q.doc_id]
            gold = resolve_gold(q, tree)
            if not gold:
                continue
            gold_set = set(gold)
            for e in trace.expansions:
                children = e.jev_children if e.jev_children is not None else e.children
                for c in children:
                    probs = c.probabilities
                    if not probs:
                        continue
                    top_levels = sorted(probs, key=lambda k: int(k))[-2:]
                    p = sum(probs[k] for k in top_levels)
                    hit = c.child_id in gold_set or any(
                        tree.is_descendant(g, c.child_id) for g in gold_set
                    )
                    predicted.append(min(max(p, 0.0), 1.0))
                    normalized.append(c.normalized)
                    outcomes.append(hit)
    if not predicted:
        return {"decisions": 0}
    curve_p = calibration_curve(predicted, outcomes)
    curve_s = calibration_curve(normalized, outcomes)
    return {
        "decisions": len(predicted),
        "definition": (
            "predicted = P(top two relevance levels); outcome = gold section in the child's subtree"
        ),
        "curve_probability": [{"predicted": p, "observed": o, "n": n} for p, o, n in curve_p],
        "ece_probability": expected_calibration_error(curve_p),
        "curve_normalized_score": [
            {"predicted": p, "observed": o, "n": n} for p, o, n in curve_s
        ],
        "ece_normalized_score": expected_calibration_error(curve_s),
    }


def write_summary_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    fields = [
        "arm",
        "name",
        "queries",
        "errors",
        "section_recall",
        "exact_match",
        "f1",
        "latency_ms_median",
        "latency_ms_p95",
        "mean_expansions",
        "escalation_rate",
        "retrieval_cost_per_query",
        "answering_cost_per_query",
        "indexing_amortized_per_query",
        "total_cost_per_query",
        "cost_per_correct_answer",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for s in summaries.values():
            if s.get("queries", 0) == 0:
                continue
            cost = s["cost_usd"]
            writer.writerow(
                {
                    "arm": s["arm"],
                    "name": s["name"],
                    "queries": s["queries"],
                    "errors": s["errors"],
                    "section_recall": s["section_recall"],
                    "exact_match": s["exact_match"],
                    "f1": s["f1"],
                    "latency_ms_median": s["latency_ms_median"],
                    "latency_ms_p95": s["latency_ms_p95"],
                    "mean_expansions": s["mean_expansions"],
                    "escalation_rate": s["escalation_rate"],
                    "retrieval_cost_per_query": cost["retrieval_per_query"],
                    "answering_cost_per_query": cost["answering_per_query"],
                    "indexing_amortized_per_query": cost["indexing_amortized_per_query"],
                    "total_cost_per_query": cost["total_per_query"],
                    "cost_per_correct_answer": cost["per_correct_answer"],
                }
            )


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------

REQUIRED_MANIFEST_KEYS = (
    "commit",
    "dependencies",
    "config",
    "dataset",
    "prompt_versions",
    "thresholds",
    "threshold_fitted_on",
    "jev",
    "llm",
    "index",
    "summary",
)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "not a git repository"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def dependency_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in (
        "nav-jev",
        "typesafe-sdk",
        "anthropic",
        "pymupdf",
        "rank-bm25",
        "numpy",
        "sentence-transformers",
    ):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def write_manifest(run_dir: Path, payload: dict[str, object]) -> None:
    """Commit, dependency versions, dataset hash, resolved Jev model ID, prompt version,
    thresholds with their provenance, per-query tokens, latency and cost, and the count
    of documents whose structure was LLM-inferred.

    A number without one of these is not reportable. See docs/eval-protocol.md.
    """
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in payload]
    if missing:
        raise ValueError(f"manifest is missing {missing}")
    dataset = payload["dataset"]
    if not isinstance(dataset, dict) or not dataset.get("content_hash"):
        raise ValueError("manifest dataset entry needs a content_hash")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
