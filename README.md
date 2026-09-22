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

## Install

```bash
pip install git+https://github.com/silvaan/nav-jev.git
```

Python 3.11+. Two keys, in the environment or in a `.env` file where you run it:

| Variable | Used for |
|----------|----------|
| `TYPESAFE_API_KEY` | Jev, the model that walks the tree |
| `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` | the summaries written once per section at index time (`gpt-*` or `claude-*` models) |

## Use

```python
from navjev import Navigator

nav = Navigator("index/")
nav.add("report.pdf")                       # parse + summarize; cached, so re-adding is free
hits = nav.search("What was the fiscal 2023 capital expenditure?")

for h in hits["results"]:                   # best sections first
    print(f"{h['score']:.2f}  {' > '.join(h['path'])}")
    print(h["text"][:200])
```

`add` takes Markdown, PDFs with an outline, or DOCX, and any number of them: `search`
looks through every document in the index unless you pass `doc=`. Each hit carries the
section's `title`, `path`, `pages` (for PDFs), `score` and full `text`; the dict also
reports `cost_usd` and `no_answer`, which is true when the question is about nothing in
the index. Pass `detail=True` to get every decision the walk made. Every method has an
async twin (`a_add`, `a_search`), and `nav.usage` shows what has been spent.

The same two steps from the shell:

```bash
nav-jev add report.pdf
nav-jev ask "What was the fiscal 2023 capital expenditure?" --detail
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
cost $0.00013
```

Each block is one Jev request: the section being looked at, how likely each child is to
hold the answer, and `->` on the ones that were opened. `--json` prints the raw result,
`--max-spend-usd` stops a run once it has spent that much.

## How it decides

The walk is a beam search. At each section, one request asks Jev to score every child
("how likely is this section to contain the evidence?") on a four-level scale, and
whether the current section already answers the question. Children above a threshold
enter the next round; if none does, the search backs up and tries the parent's other
children. Everything Jev is asked, and every threshold, is in one file you can read and
change: [`src/navjev/traverse/questions.py`](src/navjev/traverse/questions.py). The
full policy is in [`docs/algorithm.md`](docs/algorithm.md).

Files with no usable structure (a PDF without bookmarks, a plain `.txt`) can have an
outline inferred by an LLM with `allow_llm_structure=True` (`--allow-llm-structure`);
the index tags them so you know the tree was guessed rather than read.

## Options

`Navigator(index_dir, summarizer="gpt-4.1-mini", jev_model="jev-latest",
max_spend_usd=None, thresholds=Thresholds(), allow_llm_structure=False)`. The
thresholds are working defaults, not tuned ones; the first things to adjust for your own
documents are `tau_expand` (how relevant a child must look to be opened) and `tau_stop`
(how sure Jev must be that a section answers the question by itself).

## Development

```bash
git clone git@github.com:silvaan/nav-jev.git && cd nav-jev
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest              # offline, no keys needed; `pytest --live` spends a few cents
```

## License

MIT.
