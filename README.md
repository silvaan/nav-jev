# nav-jev

**Ask questions to long documents by walking their table of contents.**

nav-jev turns a document into a tree of sections, gives each section a one-line summary,
and answers a question by opening only the branches that look relevant. The branch
decisions are made by [TypeSafe's Jev](https://docs.typesafe.ai), a small model that
returns calibrated probabilities instead of generating text, so a query costs a handful
of cheap calls and every step is explainable: you see which sections were considered,
which were opened, and why.

No embeddings, no vector database. The index is the document's own structure plus the
summaries, stored as plain JSON.

```
                 ┌─ Business ──────────── Segments
  Annual report ─┼─ Financial Statements ─┬─ Cash Flow      ◀── "What was the 2023 capex?"
                 │                        └─ Balance Sheet
                 └─ Executive Compensation
```

---

## Install

Python 3.11+.

```bash
pip install git+https://github.com/silvaan/nav-jev.git
```

Keys, in the environment or in a `.env` file in the working directory (the CLI reads it;
from Python, call `navjev.env.load_dotenv()`):

| Variable | Used for |
|----------|----------|
| `TYPESAFE_API_KEY` | Jev, the model that walks the tree |
| `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` | writing the section summaries at index time (`gpt-*` or `claude-*` models) |

Every command that spends money takes a `--max-spend-usd` cap that defaults to zero, so
nothing is charged by accident.

## Use

**1. Index a document** (Markdown, PDF with an outline, or DOCX):

```bash
nav-jev build report.pdf --out index/report --summarizer gpt-4.1-mini --max-spend-usd 1
```

This parses the headings, writes one summary per section, and saves the tree to
`index/report/docs/report.json`. Summaries are cached, so re-indexing an unchanged
document is free.

**2. Ask it something:**

```bash
nav-jev ask index/report "What was the fiscal 2023 capital expenditure?" --max-spend-usd 0.1
```

```
Acme Corp Annual Report 2023  stop=0.05 off_topic=0.07  model=jev-1.13.0
     Business  score=0.08 conf=0.76
  -> Financial Statements  score=0.77 conf=0.69
     Executive Compensation  score=0.00 conf=1.00
  Financial Statements  stop=0.07  model=jev-1.13.0
    -> Cash Flow  score=1.00 conf=1.00
       Balance Sheet  score=0.09 conf=0.73

[1.00] Acme Corp Annual Report 2023 > Financial Statements > Cash Flow
```

Each block is one Jev request: the section being looked at, how likely each child is to
hold the answer, and `->` on the ones that were opened. The last line is the section to
read. A question the document does not cover stops at the root with `no answer` instead
of a forced guess.

**3. From Python**, to feed the selected sections to your own model or UI:

```python
import asyncio
from pathlib import Path

from navjev.build.index import Index
from navjev.env import load_dotenv
from navjev.jev import JevClient
from navjev.traverse.beam import BeamSearch, JevPolicy
from navjev.traverse.questions import Thresholds

load_dotenv()
index = Index(Path("index/report"))
tree = index.load_tree("report")

thresholds = Thresholds().without_fallback()
search = BeamSearch(JevPolicy(JevClient(max_spend_usd=0.1), thresholds), thresholds)
result = asyncio.run(search.retrieve(tree, "What was the fiscal 2023 capital expenditure?"))

for node in result.nodes:            # best sections first
    print(" > ".join(tree.path_to(node.id)))
    print(node.text[:300])
print(result.trace.as_path_text(tree))   # the same explanation the CLI prints
```

`result.trace` holds every decision with its probability and confidence, and
`result.cost_usd` what the query cost.

## How it decides

The policy is a beam search. At each section, one request asks Jev to score every child
("how likely is this section to contain the evidence?") on a four-level scale, and
whether the current section already answers the question. Children above a threshold
enter the next round; if none does, the search backs up and tries the parent's other
children. Everything Jev is asked, and every threshold, is in one file you can read and
change: [`src/navjev/traverse/questions.py`](src/navjev/traverse/questions.py).

Documents with no usable structure (a PDF without bookmarks, a plain `.txt`) can have an
outline inferred by an LLM with `--allow-llm-structure`; the index is tagged so you know
the tree was guessed rather than read.

## Evaluating it

The repo also ships a benchmark harness that compares this policy against BM25, dense
retrieval with a reranker, and an LLM making the same branch decisions, on FinanceBench,
QASPER and NanoHotpotQA. No numbers are published yet; when they are, they will live in
[`docs/findings.md`](docs/findings.md) with a manifest that reproduces them.

```bash
pip install "nav-jev[eval] @ git+https://github.com/silvaan/nav-jev.git"
nav-jev dataset freeze qasper --max-queries 150
nav-jev fit configs/qasper.yaml
nav-jev bench configs/qasper.yaml
nav-jev report results/<run>
```

## Development

```bash
git clone git@github.com:silvaan/nav-jev.git && cd nav-jev
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest              # offline, no keys needed
```

## License

MIT.
