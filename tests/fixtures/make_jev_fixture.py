"""The request behind jev_fixture.jsonl.

`jev_fixture.jsonl` was recorded from the live API (jev-1.13.0) by `tests/test_live.py`
against exactly this STATE and these QUESTIONS. Running this script rewrites the fixture
with the hand-written RESPONSE below, which follows the documented wire shape but is not
a real answer; do that only to bootstrap after changing the request, then re-record with
`pytest --live`.
"""

from __future__ import annotations

import json
from pathlib import Path

from navjev.jev import Noul, Score, request_key

OUT = Path(__file__).with_name("jev_fixture.jsonl")

STATE = {
    "query": "What was the fiscal 2023 capital expenditure?",
    "document_title": "Acme Corp 10-K 2023",
    "current_section": {"title": "Acme Corp 10-K 2023", "summary": "Annual report."},
    "path": ["Acme Corp 10-K 2023"],
    "children": [
        {"title": "Business", "summary": "Describes the company's segments and products."},
        {
            "title": "Financial Statements",
            "summary": "Consolidated statements including cash flows and capex by year.",
        },
        {"title": "Executive Compensation", "summary": "Pay policy for named officers."},
    ],
}

LEVELS = [
    "The section is about a different subject from the query.",
    "The section shares a topic but not the specific fact.",
    "The section plausibly holds part of what is needed.",
    "The section is stated to contain the exact subject of the query.",
]

QUESTIONS = {
    "child_0": Score(
        instructions="How likely is `children[0]` to hold the evidence?", criteria=LEVELS
    ),
    "child_1": Score(
        instructions="How likely is `children[1]` to hold the evidence?", criteria=LEVELS
    ),
    "child_2": Score(
        instructions="How likely is `children[2]` to hold the evidence?", criteria=LEVELS
    ),
    "stop_here": Noul(
        instructions="Does `current_section` already contain what `query` asks for?"
    ),
    "off_topic": Noul(instructions="Is `query` unrelated to this document?"),
}

LEGEND = {str(i): text for i, text in enumerate(LEVELS)}

RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "child_0": {
            "type": "score",
            "score": 0.9,
            "legend": LEGEND,
            "probabilities": {"0": 0.3, "1": 0.5, "2": 0.2, "3": 0.0},
            "confidence": 0.55,
        },
        "child_1": {
            "type": "score",
            "score": 2.7,
            "legend": LEGEND,
            "probabilities": {"0": 0.0, "1": 0.05, "2": 0.2, "3": 0.75},
            "confidence": 0.8,
        },
        "child_2": {
            "type": "score",
            "score": 0.1,
            "legend": LEGEND,
            "probabilities": {"0": 0.9, "1": 0.1, "2": 0.0, "3": 0.0},
            "confidence": 0.9,
        },
        "stop_here": {"type": "noul", "noul": 0.08},
        "off_topic": {"type": "noul", "noul": 0.03},
    },
    "usage": {"input_tokens": 412, "output_tokens": 62},
}


def main() -> None:
    wire = {qid: q.to_wire() for qid, q in QUESTIONS.items()}
    entry = {
        "key": request_key(STATE, wire),
        "request": {"state": STATE, "questions": wire},
        "response": RESPONSE,
    }
    OUT.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
