"""Client wrapper: request shaping, normalization, budget, replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from navjev.jev import (
    MAX_SCORE_LEVELS,
    Answer,
    Choice,
    JevBudgetExceeded,
    JevClient,
    JevError,
    JevFixtureMiss,
    JevResponse,
    Noul,
    RecordedJevClient,
    Score,
    request_key,
)
from tests.fixtures.make_jev_fixture import QUESTIONS, RESPONSE, STATE


@pytest.mark.parametrize(
    ("raw", "levels", "expected"),
    [
        (0.0, 2, 0.0),
        (1.0, 2, 1.0),
        (1.5, 4, 0.5),
        (3.0, 4, 1.0),
        (4.5, 10, 0.5),
        (9.0, 10, 1.0),
    ],
)
def test_score_normalization_divides_by_top_level_index(
    raw: float, levels: int, expected: float
) -> None:
    answer = Answer("q", "score", raw, {}, 0.9)
    assert answer.normalized(levels) == pytest.approx(expected)


def test_same_raw_score_differs_across_scales() -> None:
    # 1.5 is the top of a 2-level scale... no, it is past it, but on a 4-level scale it is
    # the midpoint. Comparing raw scores across scales is the easiest bug to ship.
    a = Answer("q", "score", 1.5, {}, 0.9)
    assert a.normalized(4) == pytest.approx(0.5)
    assert a.normalized(7) == pytest.approx(0.25)


def test_normalized_rejects_non_score() -> None:
    with pytest.raises(JevError):
        Answer("q", "noul", 0.5, None, None).normalized(4)


def test_score_wire_shape_and_level_bounds() -> None:
    wire = Score("q", ["a", {"what": "b", "examples": ["x"]}]).to_wire()
    assert wire == {
        "type": "score",
        "instructions": "q",
        "criteria": ["a", {"what": "b", "examples": ["x"]}],
    }
    with pytest.raises(JevError):
        Score("q", ["only one"]).to_wire()
    with pytest.raises(JevError):
        Score("q", [str(i) for i in range(MAX_SCORE_LEVELS + 1)]).to_wire()


def test_choice_and_noul_wire_shapes() -> None:
    assert Noul("q").to_wire() == {"type": "noul", "instructions": "q"}
    assert Noul("q", {"true": "t", "false": "f"}).to_wire()["criteria"] == {
        "true": "t",
        "false": "f",
    }
    assert Choice("q", {"a": None, "b": "desc"}).to_wire()["criteria"] == {
        "a": None,
        "b": "desc",
    }
    with pytest.raises(JevError):
        Choice("q", {str(i): None for i in range(256)}).to_wire()


def test_cost_uses_rate_from_caller() -> None:
    r = JevResponse("jev-1.13.0", {}, 1_000_000, 0, 1.0)
    assert r.cost_usd(0.042) == pytest.approx(0.042)
    assert r.cost_usd(1.0) == pytest.approx(1.0)


async def test_recorded_client_replays_fixture(fixtures_dir: Path) -> None:
    client = RecordedJevClient(fixtures_dir / "jev_fixture.jsonl")
    response = await client.ask(STATE, QUESTIONS)
    assert response.model == "jev-1.13.0"
    assert response.input_tokens == 762
    assert response.request_id is not None
    # Financial Statements (child_1) outranks Business and Executive Compensation for a
    # capex query, and the root is neither a stopping point nor off-topic.
    assert response.answers["child_1"].normalized(4) == pytest.approx(2.99 / 3)
    assert response.answers["child_0"].normalized(4) < 0.1
    assert response.answers["child_2"].normalized(4) == 0.0
    assert response.answers["stop_here"].probability == pytest.approx(0.16)
    assert response.answers["off_topic"].probability == pytest.approx(0.03)
    assert response.answers["child_2"].confidence == pytest.approx(1.0)
    assert len(client.calls) == 1
    assert client.usage.model_ids == ["jev-1.13.0"]
    assert client.spent_usd == pytest.approx(762 / 1e6 * 0.042)


async def test_recorded_client_miss_raises_instead_of_calling_network(
    fixtures_dir: Path,
) -> None:
    client = RecordedJevClient(fixtures_dir / "jev_fixture.jsonl")
    with pytest.raises(JevFixtureMiss):
        await client.ask({"query": "something else"}, {"q": Noul("Is it?")})


async def test_ask_many_issues_every_request() -> None:
    client = RecordedJevClient()
    for i in range(3):
        client.add(
            {"i": i},
            {"q": Noul("Is it?")},
            {
                "model": "jev-1.13.0",
                "answers": {"q": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )
    out = await client.ask_many([({"i": i}, {"q": Noul("Is it?")}) for i in range(3)])
    assert len(out) == 3 and len(client.calls) == 3


def test_request_key_is_order_independent_and_content_sensitive() -> None:
    q = {"a": Noul("x"), "b": Noul("y")}
    k1 = request_key({"s": 1, "t": 2}, q)
    k2 = request_key({"t": 2, "s": 1}, {"b": Noul("y"), "a": Noul("x")})
    assert k1 == k2
    assert request_key({"s": 1}, q) != k1


async def test_zero_spend_cap_blocks_every_paid_call() -> None:
    client = JevClient(api_key="unused", max_spend_usd=0.0)
    with pytest.raises(JevBudgetExceeded):
        await client.ask("state", {"q": Noul("Is it?")})


async def test_spend_cap_stops_further_calls() -> None:
    class Scripted(RecordedJevClient):
        pass

    client = Scripted(rate_per_million=1_000_000.0)  # $1 per token, so a call costs $10
    client.max_spend_usd = 15.0
    body = {
        "model": "jev-1.13.0",
        "answers": {"q": {"type": "noul", "noul": 0.5}},
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }
    client.add("s", {"q": Noul("Is it?")}, body)
    await client.ask("s", {"q": Noul("Is it?")})
    await client.ask("s", {"q": Noul("Is it?")})  # spent $20 > $15 after this one
    with pytest.raises(JevBudgetExceeded):
        await client.ask("s", {"q": Noul("Is it?")})


async def test_oversized_request_fails_locally() -> None:
    client = RecordedJevClient()
    with pytest.raises(JevError, match="over the"):
        await client.ask("x" * 200_000, {"q": Noul("Is it?")})


async def test_empty_questions_rejected() -> None:
    client = RecordedJevClient()
    with pytest.raises(JevError):
        await client.ask("state", {})


def test_answer_from_wire_rejects_unknown_type() -> None:
    with pytest.raises(JevError):
        Answer.from_wire("q", {"type": "vector"})


def test_fixture_was_recorded_for_the_generator_request() -> None:
    # Guards against the checked-in recording drifting from the request that made it.
    client = RecordedJevClient(Path(__file__).parent / "fixtures" / "jev_fixture.jsonl")
    wire = {qid: q.to_wire() for qid, q in QUESTIONS.items()}
    recorded = client._responses[request_key(STATE, wire)]
    assert set(recorded["answers"]) == set(RESPONSE["answers"])
    assert recorded["model"].startswith("jev-") and recorded["model"] != "jev-latest"
