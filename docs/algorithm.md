# Traversal algorithm

## Notation

A document is a tree `T` of nodes. Node `n` carries `title`, `summary`, `page_span`,
`char_span`, `depth`, `children`, and a reference to its full text, which traversal does
not read. A query is a string `q`. Retrieval returns an ordered list of leaf or internal
nodes whose full text is then handed to an answering model, plus a trace of every decision.

## Indexing, once per document

Indexing uses no Jev. The parser recovers the native hierarchy: ATX headings for Markdown,
the outline or bookmark tree for PDF, heading styles for DOCX. Documents with no usable
structure can fall back to a heading-detection pass by an LLM, flagged in the index
metadata, because a guessed outline is a different thing from the author's own.

Each node gets a summary of at most 60 words written by a cheap LLM from the node's own
text plus its ancestors' titles, so a subsection summary is not ambiguous when read out
of context. Summaries are cached on disk under the SHA-256 of the node text plus the
summarizer's model ID and prompt version, which makes reindexing free and makes a changed
prompt a cache miss rather than a silent inconsistency.

## Retrieval

Beam search from the root. The beam holds `(node, score)` pairs. At each step, every node
in the beam is expanded in its own Jev request, and the requests for one level go out
concurrently.

For a node `n` with children `c_1 … c_k`, the state is an object:

```json
{
  "query": "...",
  "document_title": "...",
  "current_section": {"title": "...", "summary": "..."},
  "path": ["Document title", "Chapter 3", "3.2 Revenue"],
  "children": [{"title": "...", "summary": "..."}, ...]
}
```

The questions, all in one request:

- `child_i` for each child, a `Score` over evidence likelihood. The criteria describe
  situations, from "nothing in this section relates to the query" up to "this section
  almost certainly contains the answer or the data the query asks for". Instructions point
  at `` `children[i]` `` explicitly.
- `stop_here`, a `Noul`: does `current_section` already contain what the query asks for,
  such that opening a subsection is unnecessary.
- `off_topic`, a `Noul`: is the query unrelated to this document. Evaluated only at the
  root, where it gives the system an honest "no answer here" rather than a forced path.

Children with normalized score above `tau_expand` enter the next beam, capped at width
`b`. Nodes whose `stop_here` exceeds `tau_stop` are emitted as results. Search halts on an
empty beam, at `max_depth`, or at a per-query call budget. When no child clears
`tau_expand` at a node that is not itself a result, the node is recorded as a dead end and
its parent's remaining siblings become eligible, which keeps a single bad decision at
depth 2 from losing the document.

Normalization matters and is a code concern, not a model one. A raw `score` is the
probability-weighted mean of level indices, so it is divided by the top level index before
any comparison against a threshold or across questions with different scale lengths.

## Fallback (off by default)

A node expansion whose `confidence` falls below `tau_llm` on the decisive questions can be
re-decided by an LLM given the same state. `Navigator` never enables it; it exists for
anyone measuring the policy who wants a safety net, and the trace marks every escalation.
A policy that escalates most of the time has not replaced the LLM, it has added a call in
front of it.

## Where the saving comes from

For a tree of branching factor `k` and depth `d`, an LLM walking the tree makes `O(b·d)`
sequential reasoning calls, each carrying the sibling summaries. Jev traversal makes the
same number of requests, but each is a fan-out of `k+2` independent questions answered in
one call by a model that does not generate text. The token counts are comparable, since
both send the same state; the price per token and the latency per call are not.

## Known weak spots

Deep narrow trees, where an early wrong turn is unrecoverable and the dead-end rule is
load-bearing. Documents whose section titles are uninformative, where the summary quality
dominates. Questions needing evidence from two distant sections, which the beam width has
to accommodate. Adversarial text inside a document that addresses the model directly:
TypeSafe states that state is treated as data, but steering text can still move an
answer.
