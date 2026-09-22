# Instructions for the coding agent

nav-jev is a small tool: it indexes a document as its section tree and uses TypeSafe's
Jev to decide which sections to open for a question. Keep it small. The public surface
is `Navigator` (`navjev/navigator.py`) and the two CLI commands; everything else is an
implementation detail that a user should not need to touch. Read `docs/algorithm.md`
before changing the traversal.

## Hard rules

1. **Never let Jev generate or extract text.** Jev returns a probability over options you
   supplied. Every summary and every title comes from the parser or from an LLM. If a
   task feels like "extract X", rewrite it as "here are the candidates for X, which one
   is it".
2. **No arithmetic, counting, or date comparison inside a question.** Jev is documented as
   unreliable at all three. Compute in Python and pass the result in as state.
3. **One judgment per question.** Do not ask "is this node relevant and should we stop".
   That is two questions.
4. **Fan out, do not loop.** All children of a node go in one request, as independent
   questions sharing one state. Sequential per-child calls are a bug, not a style choice.
   The stop test rides along in the same request.
5. **Every question and every threshold lives in `src/navjev/traverse/questions.py`.**
   No prompt text scattered across modules. A reader must go through one file to know
   what the system asks.
6. **Record the model ID.** The response reports the versioned ID that answered; it goes
   in every trace. Thresholds tuned against one version are not valid against another.
7. **Stub honestly.** If something is unimplemented, `raise NotImplementedError`. Never
   return a plausible fake value.
8. **No benchmark code here.** The evaluation harness is a separate project. Do not add
   datasets, baselines, metrics or manifests to this repo.

## The Jev API, verbatim

Endpoint `POST https://api.typesafe.ai/v1/systemone`, bearer auth, key in
`TYPESAFE_API_KEY`. Python SDK is `typesafe-sdk` (Python 3.10+), exposing
`TypeSafeClient`, `AsyncTypeSafeClient`, and the `Noul`, `Choice`, `Score` question
classes. The live docs at https://docs.typesafe.ai/llms.txt are the source of truth; the
`typesafe` agent skill points at them.

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
429, 529. Retry 429 and 529 with exponential backoff (the SDK's retry policy does).

Pricing at the time of writing is $0.042 per million input tokens, output free. Cost
accounting multiplies reported `usage.input_tokens` by a rate passed in by the caller;
the rate is never hardcoded.

## Layout

```
src/navjev/
  navigator.py   the facade: Navigator.add / search, sync and async
  cli.py         nav-jev add, nav-jev ask
  jev.py         Jev client: fan-out, spend cap, recording and replay
  llm.py         the one module that calls a text LLM (OpenAI or Anthropic by model name)
  build/         parsers, summaries, on-disk index
  traverse/      questions.py, beam.py
  types.py       TreeNode, DocumentTree, Expansion, Trace, RetrievalResult
```

## Tooling

Python 3.11+, `pytest`, `ruff`, `mypy` in strict mode on `src/`. Async in the retrieval
path, with sync wrappers on the facade. No network calls in the unit suite; `pytest
--live` runs the few paid tests, capped at cents. The Jev fixture in `tests/fixtures/` is
a real recording; re-record it with `--live` after changing a question.

## Things that look like good ideas and are not

- Asking Jev one `Choice` over all children instead of one `Score` each. A `Choice` forces
  a single winner, which forbids the beam and hides the case where two branches both hold
  evidence. Score per child, decide in code.
- Putting the full node text in the state during traversal. The summary and title are what
  the decision needs; the full text blows the budget and, per TypeSafe's own guidance,
  accuracy falls as the state fills with irrelevant content.
- Feeding the whole tree in one request. It defeats the purpose, which is to read only the
  branches worth opening.
- Treating a `score` of 1.4 as a magnitude. Levels are weakly calibrated against each
  other. Use it to pass a threshold or to rank, never to interpolate.
- Growing the public surface. If a feature needs a third class in the README, it probably
  belongs in `Navigator` or nowhere.
