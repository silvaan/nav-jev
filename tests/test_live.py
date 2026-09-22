"""Paid tests, run with `pytest --live`. Capped at a few cents. The Jev test appends its
real response to the offline fixture, so re-running it after a request change refreshes
what the offline suite replays."""

from __future__ import annotations

from pathlib import Path

import pytest

from navjev.jev import JevClient
from tests.fixtures.make_jev_fixture import QUESTIONS, STATE

pytestmark = pytest.mark.live
LIVE_CAP_USD = 0.05


@pytest.fixture
def live_client() -> JevClient:
    return JevClient(
        max_spend_usd=LIVE_CAP_USD,
        record_to=Path(__file__).parent / "fixtures" / "jev_fixture.jsonl",
    )


async def test_one_expansion_request_has_documented_shape(live_client: JevClient) -> None:
    response = await live_client.ask(STATE, QUESTIONS)
    assert response.model.startswith("jev-") and response.model != "jev-latest"
    assert response.input_tokens > 0
    for i in range(3):
        a = response.answers[f"child_{i}"]
        assert a.type == "score" and 0.0 <= a.normalized(4) <= 1.0
        assert a.confidence is not None and 0.0 <= a.confidence <= 1.0
        assert a.probabilities is not None and abs(sum(a.probabilities.values()) - 1) < 1e-3
    assert 0.0 <= response.answers["stop_here"].probability <= 1.0
    assert 0.0 <= response.answers["off_topic"].probability <= 1.0
    # The financial statements child should outrank executive compensation for a capex query.
    assert response.answers["child_1"].normalized(4) > response.answers["child_2"].normalized(4)
    assert live_client.spent_usd > 0
    await live_client.aclose()
