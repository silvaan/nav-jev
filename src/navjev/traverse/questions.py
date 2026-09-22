"""Every question and every threshold in the retrieval policy.

This file is the audit surface. A reader goes through it alone to know what the system
asks and when it acts. Nothing that shapes a decision lives anywhere else: the state
layout the questions point into, the Jev questions, the decision rules, the thresholds,
and the prompt the optional LLM fallback receives.

Bump PROMPT_VERSION on any change to a question or a criterion: thresholds tuned against
one wording do not carry over to another.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from navjev.jev import Noul, Score
from navjev.types import ChildDecision, DocumentTree, NodeId

PROMPT_VERSION = "v1"


# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------


def build_state(tree: DocumentTree, node_id: NodeId, query: str) -> dict[str, Any]:
    """State object for one expansion. Keys are referenced by backticked paths in the
    questions below, so this function and the questions change together.

    Only titles and summaries are sent. Full node text never enters the state: it blows
    the token budget and, per TypeSafe's guidance, accuracy falls as the state fills with
    irrelevant content.
    """
    node = tree.nodes[node_id]
    return {
        "query": query,
        "document_title": tree.title,
        "current_section": {"title": node.title, "summary": node.summary or ""},
        "path": tree.path_to(node_id),
        "children": [
            {"title": c.title, "summary": c.summary or ""} for c in tree.children_of(node_id)
        ],
    }


# --------------------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------------------


def child_relevance(index: int) -> Score:
    """Evidence likelihood for one child section.

    One Score per child, all in the same request. Not a single Choice over all children:
    a Choice forces one winner, which forbids the beam and hides the case where two
    branches both hold evidence.

    Levels describe situations rather than degrees, and are read independently of one
    another, so each has to stand alone.
    """
    return Score(
        instructions=(
            f"Considering `query`, how likely is it that the section described in "
            f"`children[{index}]` contains the evidence needed to answer it? "
            f"Judge the section itself, not its neighbours, and use `path` only to "
            f"understand where the section sits in the document."
        ),
        criteria=[
            {
                "what": (
                    "The section is about a different subject from the query, and "
                    "nothing in its title or summary bears on what the query asks."
                ),
                "examples": [
                    (
                        "Query asks about capital expenditure; the section describes "
                        "executive compensation policy."
                    )
                ],
            },
            {
                "what": (
                    "The section shares a general topic with the query but does not "
                    "appear to hold the specific fact, figure, or statement requested."
                ),
                "examples": [
                    (
                        "Query asks for the fiscal 2023 figure; the section is a narrative "
                        "overview of the same business line with no figures."
                    )
                ],
            },
            {
                "what": (
                    "The section plausibly holds part of what the query needs, or leads "
                    "to subsections that would, but the summary does not confirm it."
                ),
                "examples": [
                    (
                        "Query asks for a specific line item; the section is the notes to "
                        "the financial statements, which contain many such items."
                    )
                ],
            },
            {
                "what": (
                    "The section is stated to contain the exact subject of the query, "
                    "so the answer is almost certainly inside it or inside its "
                    "immediate subsections."
                ),
                "examples": [
                    (
                        "Query asks for the fiscal 2023 capital expenditure; the summary "
                        "says the section reports capital expenditure by year."
                    )
                ],
            },
        ],
    )


RELEVANCE_LEVELS = child_relevance(0).levels
"""Number of levels on the child scale. Normalization divides by RELEVANCE_LEVELS - 1."""


STOP_HERE = Noul(
    instructions=(
        "Does the section described in `current_section` already contain what `query` "
        "asks for, so that opening one of its subsections would not be needed?"
    ),
    criteria={
        "true": (
            "The summary states that this section reports the specific fact, figure, or "
            "statement the query asks for."
        ),
        "false": (
            "The section is an introduction, an overview, or a container whose subsections "
            "hold the detail, or it does not mention what the query asks for at all."
        ),
    },
)

OFF_TOPIC = Noul(
    instructions=(
        "Is `query` about a subject this document does not cover at all, judging by "
        "`document_title` and the sections listed in `children`?"
    ),
    criteria={
        "true": "The document is about an unrelated subject, entity, or period.",
        "false": (
            "The document covers the subject of the query, even if the specific answer "
            "may not be present."
        ),
    },
)
"""Asked only at the root. Gives the system an honest 'not in this document' instead of a
forced path, which similarity search cannot express."""


CHILD_QUESTION_PREFIX = "child_"
STOP_QUESTION_ID = "stop_here"
OFF_TOPIC_QUESTION_ID = "off_topic"


def expansion_questions(child_count: int, at_root: bool) -> dict[str, Noul | Score]:
    """Every question for one node, in one request: k child scores, the stop test, and
    at the root the off-topic test. Question ids are for code only."""
    questions: dict[str, Noul | Score] = {
        f"{CHILD_QUESTION_PREFIX}{i}": child_relevance(i) for i in range(child_count)
    }
    questions[STOP_QUESTION_ID] = STOP_HERE
    if at_root:
        questions[OFF_TOPIC_QUESTION_ID] = OFF_TOPIC
    return questions


# --------------------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """All values are on normalized scores in [0, 1] or on Noul probabilities in [0, 1].

    The defaults are working values for the current questions, not tuned ones. Anyone
    measuring the policy on their own documents should expect to adjust `tau_expand` and
    `tau_stop` first.
    """

    tau_expand: float = 0.55
    """Normalized child score above which a child enters the next beam."""

    tau_stop: float = 0.70
    """Noul probability above which the current node is emitted as a result."""

    tau_llm: float = 0.0
    """Confidence below which the expansion is re-decided by an LLM. Zero, the default,
    disables the fallback; the Navigator never enables it."""

    tau_off_topic: float = 0.85
    """Deliberately high. Declaring a document irrelevant ends the search, so it needs
    more evidence than an ordinary branch decision."""

    beam_width: int = 3
    max_depth: int = 6
    max_expansions: int = 24
    """Per-query call budget. A runaway traversal is a cost bug, so it fails loudly."""

    @property
    def fallback_enabled(self) -> bool:
        return self.tau_llm > 0.0

    def without_fallback(self) -> Thresholds:
        return replace(self, tau_llm=0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_THRESHOLDS = Thresholds()


# --------------------------------------------------------------------------------------
# Decision rules. Pure functions over answers; the beam applies them.
# --------------------------------------------------------------------------------------


def select_children(children: Sequence[ChildDecision], thresholds: Thresholds) -> list[int]:
    """Indices of the children that enter the next beam: those above `tau_expand`, best
    first, capped at `beam_width`. Ties keep document order."""
    ranked = sorted(
        (i for i, c in enumerate(children) if c.normalized >= thresholds.tau_expand),
        key=lambda i: (-children[i].normalized, i),
    )
    return ranked[: thresholds.beam_width]


def should_stop(stop_here: float, thresholds: Thresholds) -> bool:
    return stop_here >= thresholds.tau_stop


def is_off_topic(off_topic: float | None, thresholds: Thresholds) -> bool:
    return off_topic is not None and off_topic >= thresholds.tau_off_topic


def needs_escalation(children: Sequence[ChildDecision], thresholds: Thresholds) -> bool:
    """The decisive questions are the child scores that determine what gets opened: the
    children selected for the beam, or, when none clears the floor, the single best
    child. If any of those was answered with confidence below `tau_llm`, the expansion
    is re-decided by an LLM. Other children can be uncertain without consequence."""
    if not thresholds.fallback_enabled or not children:
        return False
    decisive = select_children(children, thresholds)
    if not decisive:
        decisive = [max(range(len(children)), key=lambda i: children[i].normalized)]
    return any(children[i].confidence < thresholds.tau_llm for i in decisive)


# --------------------------------------------------------------------------------------
# The prompt for the optional LLM fallback
#
# It receives exactly the state Jev receives and returns the same shape: one relevance
# level per child on the same scale, and a stop probability, so a trace reads the same
# whichever model decided.
# --------------------------------------------------------------------------------------

LLM_TRAVERSAL_SYSTEM = (
    "You navigate a document's table of contents to find where a question is answered. "
    "You see the question, the current section, its position in the document, and its "
    "child sections as titles and short summaries. For every child, rate how likely it is "
    "to contain the evidence needed, on the scale given. Then say whether the current "
    "section itself already contains what the question asks for, as a probability. "
    "Judge only from the titles and summaries; do not assume content they do not state."
)


def llm_traversal_prompt(state: dict[str, Any]) -> str:
    levels = child_relevance(0).criteria
    scale = "\n".join(
        f"  {i}: {c['what'] if isinstance(c, dict) else c}" for i, c in enumerate(levels)
    )
    return (
        f"Relevance scale for each child (integer level):\n{scale}\n\n"
        f"State:\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
        f"Return one relevance level per child, in order, and stop_here as a "
        f"probability from 0 to 1 that `current_section` already contains the answer."
    )


def llm_traversal_schema(child_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "children": {
                "type": "array",
                "minItems": child_count,
                "maxItems": child_count,
                "items": {"type": "integer", "minimum": 0, "maximum": RELEVANCE_LEVELS - 1},
            },
            "stop_here": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["children", "stop_here"],
        "additionalProperties": False,
    }
