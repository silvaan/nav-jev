# nav-jev

**Retrieval that walks a document's table of contents, with a System One model making
each turn instead of an LLM.**

Long documents already have a structure: chapters, sections, subsections. nav-jev builds
that tree once, gives every node a one-line summary, and answers a question by walking
down the branches that look relevant. The novelty is *who decides which branch to open*:
[TypeSafe's Jev](https://docs.typesafe.ai), a model that answers a batch of
yes/no and scoring questions with calibrated probabilities in one fast call, instead of
an LLM that reasons at every level.

> **Status:** implemented and running end to end; **no benchmark numbers yet**.
> Nothing here should be cited as evidence until `results/` holds a run manifest.

---

## How it works

```
                 ┌─ Business ──────────── Segments
  Annual report ─┼─ Financial Statements ─┬─ Cash Flow      ◀── answer is here
                 │                        └─ Balance Sheet
                 └─ Executive Compensation
```

1. **Index once.** A parser recovers the native hierarchy (Markdown headings, PDF
   outline, DOCX styles) and a cheap LLM writes a summary of at most 60 words per node.
   Summaries are cached on disk by content hash.
2. **Walk per query.** Starting at the root, one Jev request scores every child section
   at once ("how likely is this section to hold the evidence?") and asks whether the
   current section already answers the question. Children above a fitted threshold enter
   a beam; the walk continues until the beam empties.
3. **Answer.** The selected sections' full text goes to an answering model. Every
   decision, with its probability and confidence, is kept in a trace.

The whole policy, every question, threshold and prompt, lives in one file:
[`src/navjev/traverse/questions.py`](src/navjev/traverse/questions.py).

### What a query looks like

```
$ nav-jev ask index/sample "What was the fiscal 2023 capital expenditure?"

Acme Corp Annual Report 2023  stop=0.05 off_topic=0.07  model=jev-1.13.0
     Business  score=0.08 conf=0.76
  -> Financial Statements  score=0.77 conf=0.69
     Executive Compensation  score=0.00 conf=1.00
  Financial Statements  stop=0.07  model=jev-1.13.0
    -> Cash Flow  score=1.00 conf=1.00
       Balance Sheet  score=0.09 conf=0.73

[1.00] Acme Corp Annual Report 2023 > Financial Statements > Cash Flow
```

Each block is one request. `->` marks the children that entered the beam. A question the
document cannot answer stops at the root with `no answer`, from the `off_topic` test,
rather than a forced path.

---

## The claim under test

**At equal or near-equal section recall, Jev traversal costs and takes an order of
magnitude less than LLM traversal.** It may also simply lose. Jev has already been
measured losing to vector retrieval as a *reranker*; traversal is a different job, a
bounded choice among a handful of siblings, repeated at every level, which is where LLM
latency compounds. The benchmark exists to find out, and reports a negative result the
same way as a positive one.

Five arms, same tree, same answering model, only retrieval differs:

| Arm | Retrieval | Role |
|-----|-----------|------|
| A | BM25 over leaf sections | the floor |
| B | Dense embeddings + cross-encoder reranker | the conventional pipeline |
| C | LLM decides which branches to open | PageIndex-style, the thing being replaced |
| D | Jev decides, no fallback | the proposal |
| E | Jev decides, escalates to the LLM below a confidence threshold | the proposal with a safety net |

Datasets: FinanceBench (filings), QASPER (papers), NanoHotpotQA (multi-hop). Metrics:
section recall, answer EM/F1, latency (median, p95), cost per query and per correct
answer, escalation rate, and a calibration curve of every Jev decision. Details in
[`docs/eval-protocol.md`](docs/eval-protocol.md).

---

## Quick start

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:silvaan/nav-jev.git && cd nav-jev
uv sync --extra dev
cp .env.example .env       # TYPESAFE_API_KEY, plus OPENAI_API_KEY or ANTHROPIC_API_KEY
uv run pytest              # offline, no keys needed
```

Index one document and ask it something. Spend caps default to zero and must be raised
on purpose:

```bash
uv run nav-jev build report.pdf --out index/report --config configs/qasper.yaml --max-spend-usd 1
uv run nav-jev ask index/report "What was the fiscal 2023 capital expenditure?" --max-spend-usd 0.1
```

The text LLM is chosen per model name in the config: `gpt-*` uses OpenAI,
`claude-*` uses Anthropic.

## Running the benchmark

```bash
uv sync --extra dev --extra eval
uv run nav-jev dataset freeze qasper --max-queries 150   # data/qasper/{dev,test}, hashed
uv run nav-jev fit configs/qasper.yaml                   # thresholds fitted on dev
uv run nav-jev bench configs/qasper.yaml                 # all arms on test -> results/<run>/
uv run nav-jev report results/<run>                      # tables + calibration figure
```

`bench` refuses to run when the split is not frozen, when the thresholds were fitted on
the split being evaluated, on another dataset, or under an older prompt version, when the
Jev model ID is missing, or when a spend cap is zero. `--limit N` and a Jev model version
that differs from the fitted one mark the run *exploratory* in the manifest. A PDF with
no outline is refused unless `index.allow_llm_inferred_structure` is on, and the manifest
then counts how many documents needed it.

The manifest records the commit, dependency versions, the config verbatim, the dataset
hash, the resolved Jev model ID, prompt versions, thresholds and their provenance,
per-query tokens, latency and cost for every arm, paired-bootstrap intervals on every
difference, and the calibration curve. **A number without a manifest is not a result.**

---

## Layout

```
src/navjev/
  jev.py         Jev client: fan-out, retries, spend cap, record and replay
  llm.py         the one module that calls a text LLM
  build/         parsers, summaries, on-disk index
  traverse/      questions.py, the beam search, the LLM fallback
  baselines/     BM25, dense + reranker, LLM traversal
  eval/          frozen datasets, metrics, threshold fitting, runner, report
configs/         one config per reported run; fitted thresholds under thresholds/
docs/            algorithm.md, eval-protocol.md, findings.md
tests/           offline suite; `pytest --live` runs paid tests under a spend cap
```

## License

MIT.
