# Evaluation protocol

The protocol is fixed before any run. Changing it after seeing results means the run is
reported as exploratory and does not enter `docs/findings.md`.

## Datasets

**FinanceBench** for long single-document question answering over filings, which is where
tree retrieval is claimed to win and where the 98.7% figure that motivates this work comes
from. **QASPER** for questions over scientific papers, which have clean section structure
and answer spans annotated at the paragraph level, giving section-level recall a ground
truth. **NanoHotpotQA** as a multi-hop stress case, where a single path through one tree
cannot succeed and the beam has to carry two branches.

Each dataset loader writes a frozen split with a recorded hash. A run against an unfrozen
split is rejected by the runner.

## Arms

All arms answer with the same model, the same prompt, and the same context budget in
characters. Only retrieval differs.

- **A. BM25** over leaf sections.
- **B. Dense plus cross-encoder reranker**, the conventional pipeline.
- **C. LLM traversal** over the navjev tree, the PageIndex-style policy.
- **D. Jev traversal**, fallback disabled.
- **E. Jev traversal**, fallback enabled at the threshold fitted on dev.

C, D, and E share one index, so the difference between them is the policy alone. E is
configuration-locked until a `threshold_source: dev` line is recorded, which prevents the
threshold from being tuned on the test split.

## Metrics

Section recall at the retrieved-node count, which is the metric the policy actually
controls. Answer exact match and F1, to show whether retrieval differences survive
generation. Wall-clock latency per query, reported as median and p95, measured from the
same location for every arm in a run. Cost per query, computed from reported token usage
and a rate in the config. Escalation rate for arm E. For every Jev decision, the
probability and confidence, kept in the trace so calibration can be plotted after the fact.

Calibration deserves its own figure. RLCD's claim is that answers given 90% probability
are right about 90% of the time, and the traversal data is a natural test of it: bucket
every child decision by predicted probability and plot the observed fraction whose subtree
contained a labeled positive. If that curve is close to the diagonal, thresholds transfer
between documents and the approach is usable; if it is not, every threshold is a per-corpus
fit and the method is much less attractive. Say so either way.

## Statistics

Paired bootstrap over queries, since all arms answer the same questions. Report the
confidence interval on each difference, not only the point estimate. With fewer than a few
hundred queries, differences of one or two points in recall are not distinguishable, and
the write-up says that rather than ranking arms by a decimal.

## Cost accounting covers the whole pipeline

The Jev share of a query is small by construction, so a comparison that counts only it is
dishonest. Every run reports indexing cost amortized per query at a stated corpus size,
summarization cost, retrieval cost, the answering model's cost, and fallback cost. The
headline number is cost per correctly answered question, not cost per token.

## What would falsify the hypothesis

Arm D within noise of arm C on cost or latency; arm D losing more than a small margin of
recall to arm C; arm E escalating so often that it lands on C's cost; or a calibration
curve far enough from the diagonal that thresholds do not transfer across documents. Any
of these is the result, and `docs/findings.md` leads with it.
