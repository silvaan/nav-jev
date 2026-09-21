# Instructions for the coding agent

This repository is a research artifact. Its value is the measurement, not the code, so
correctness of the evaluation matters more than features. Read `docs/algorithm.md` and
`docs/eval-protocol.md` before writing anything.

## Hard rules

1. **Never let Jev generate or extract text.** Jev returns a probability over options you
   supplied. Every summary, every answer string, every node title comes from the parser or
   from an LLM. If a task feels like "extract X", rewrite it as "here are the candidates
   for X, which one is it".
2. **No arithmetic, counting, or date comparison inside a question.** Jev is documented as
   unreliable at all three. Compute in Python and pass the result in as state.
3. **One judgment per question.** Do not ask "is this node relevant and should we stop".
   That is two questions.
4. **Fan out, do not loop.** All children of a node go in one request, as independent
   questions sharing one state. Sequential per-child calls are a bug, not a style choice.
   The same holds for the stop test, which rides along in the same request.
5. **Every question and every threshold lives in `src/navjev/traverse/questions.py`.**
   No prompt text scattered across modules. A reviewer must read one file to audit the
   policy.
6. **Pin the model ID in results.** The response reports the versioned ID that answered.
   Log it; thresholds are only valid against the version they were fitted on.
7. **Do not report a benchmark number that a manifest in `results/` does not back.**
   No estimated costs, no extrapolated latencies, no numbers in the README that a
   `uv run nav-jev bench` invocation cannot reproduce.
8. **Stub honestly.** If something is unimplemented, `raise NotImplementedError`. Never
   return a plausible fake value from a function the evaluation will call.

## The Jev API, verbatim

Endpoint `POST https://api.typesafe.ai/v1/systemone`, bearer auth, key in
`TYPESAFE_API_KEY`. Python SDK is `typesafe-sdk` (Python 3.10+), exposing
`TypeSafeClient`, `AsyncTypeSafeClient`, and the `Noul`, `Choice`, `Score` question
classes.

Request: `model`, `state` (a string, a JSON object, or a JSON array of text), and
`questions`, a dict of question id to question. Every question has `type` and
`instructions`; `Choice` and `Score` also need `criteria`, and `Noul` takes it optionally.
Question ids are not sent to the model, so the full question goes in `instructions`.
Point at parts of an object state with backticked paths, as in `` `children[3].summary` ``.

Answer shapes:

- `noul` → `{"type": "noul", "noul": 0.93}`. No separate confidence; the probability is
  the uncertainty signal. A 0.5 means the model cannot tell, not "medium".
- `choice` → `choice`, `probabilities` over the option keys, `confidence`. Max 255 options.
- `score` → `score` (the probability-weighted mean of the level indices), `legend`,
  `probabilities`, `confidence`. `criteria` is an ordered list, low to high, of 2 to 10
  levels. Each level must describe a *situation*, not a degree. Level text may be an
  object with `what` and `examples`, and relevant examples measurably sharpen confidence.

Limits: state plus all questions share about 64k tokens; state plus the longest single
question must fit in about 32k. Text only. Errors: 401, 422 with the offending field,
429, 529. Retry 429 and 529 with exponential backoff. Rate limits during early access are
250k tokens per second and 1,200 requests per minute.

Pricing at the time of writing is $0.042 per million input tokens, output free. The cost
accounting module multiplies reported `usage.input_tokens` by a rate from the config, so
the rate is never hardcoded in the code.

Install TypeSafe's own agent skill before working on the client, since models trained on
LLM APIs invent request fields:

```bash
claude plugin marketplace add typesafe-ai/skills
claude plugin install typesafe@typesafe-ai
```

## Build order

Work in this sequence and stop at each checkpoint for review.

1. `types.py`, `jev.py`, and their tests, against a recorded-response fixture so the suite
   runs with no API key.
2. `build/`: Markdown and PDF parsers producing a `DocumentTree`, then LLM summarization
   with an on-disk cache keyed by node content hash. Checkpoint: a tree for a real
   document, inspectable as JSON.
3. `traverse/`: questions, the beam search, the trace. Checkpoint: one query end to end
   with a readable path.
4. `baselines/`: BM25, dense plus reranker, and the LLM traversal policy over the *same*
   tree, so the comparison isolates the policy rather than the index.
5. `eval/`: datasets, metrics, threshold fitting, the runner, the manifest. Checkpoint:
   a full run on a small split with a written manifest.
6. `docs/findings.md` and the README numbers, last.

## Tooling

Python 3.11+, `uv` for dependency management, `pytest`, `ruff`, `mypy` in strict mode on
`src/`. Async throughout the retrieval path, since the point of the project is latency.
Typer for the CLI. No network calls in the unit test suite; live tests sit behind a
`--live` flag and a spend cap read from the config, defaulting to zero.

## Things that look like good ideas and are not

- Asking Jev one `Choice` over all children instead of one `Score` each. A `Choice` forces
  a single winner, which forbids the beam and hides the case where two branches both hold
  evidence. Score per child, decide in code.
- Putting the full node text in the state during traversal. The summary and title are what
  the decision needs; the full text blows the budget and, per TypeSafe's own guidance,
  accuracy falls as the state fills with irrelevant content.
- Feeding the whole tree in one request. It defeats the purpose, which is to read only the
  branches worth opening, and it is exactly the cost profile the project is testing against.
- Treating a `score` of 1.4 as a magnitude. Levels are weakly calibrated against each
  other. Use it to pass a threshold or to rank, never to interpolate.
- Reusing a threshold across model versions, datasets, or question rewrites. Refit.
