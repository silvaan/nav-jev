from __future__ import annotations

from typing import Any

import pytest

from navjev.baselines.llm_traversal import LlmPolicy
from navjev.llm import ScriptedLlmClient
from navjev.traverse.beam import BeamSearch, JevPolicy, LlmFallback, TraversalBudgetExceeded
from navjev.traverse.questions import (
    OFF_TOPIC_QUESTION_ID,
    STOP_QUESTION_ID,
    Thresholds,
    build_state,
    needs_escalation,
    select_children,
)
from navjev.types import ChildDecision
from tests.helpers import ScriptedJevClient, by_title, tree_from_spec

SPEC: dict[str, Any] = {
    "title": "Doc",
    "children": [
        {"title": "A", "children": [{"title": "A1"}, {"title": "A2"}]},
        {
            "title": "B",
            "children": [{"title": "B1"}, {"title": "B2", "children": [{"title": "B2x"}]}],
        },
        {"title": "C"},
    ],
}

NO_FALLBACK = Thresholds(tau_expand=0.5, tau_stop=0.7, tau_llm=0.0, beam_width=2)


def search(
    handler: Any, thresholds: Thresholds = NO_FALLBACK
) -> tuple[BeamSearch, ScriptedJevClient]:
    client = ScriptedJevClient(handler)
    return BeamSearch(JevPolicy(client, thresholds), thresholds), client


async def test_fan_out_one_request_per_node_with_k_plus_one_questions() -> None:
    tree = tree_from_spec(SPEC)
    bs, client = search(by_title({"A": 0.9, "A1": 0.9}))
    result = await bs.retrieve(tree, "q")
    root_call = client.calls[0]
    # Root: 3 children + stop + off_topic. Question ids point at children[i].
    assert len(root_call["questions"]) == 3 + 2
    assert OFF_TOPIC_QUESTION_ID in root_call["questions"]
    assert "`children[2]`" in root_call["questions"]["child_2"]["instructions"]
    a_call = client.calls[1]
    assert a_call["state"]["current_section"]["title"] == "A"
    assert len(a_call["questions"]) == 2 + 1  # no off_topic below the root
    assert STOP_QUESTION_ID in a_call["questions"]
    assert OFF_TOPIC_QUESTION_ID not in a_call["questions"]
    # Two nodes expanded (root, A), two requests, no per-child calls.
    assert len(client.calls) == 2
    assert result.node_ids == ["A1"]
    assert result.trace.total_input_tokens > 0 and result.cost_usd > 0


async def test_state_holds_titles_and_summaries_only() -> None:
    tree = tree_from_spec(SPEC)
    state = build_state(tree, "B", "q")
    assert state["path"] == ["Doc", "B"]
    assert state["children"] == [
        {"title": "B1", "summary": "summary of B1"},
        {"title": "B2", "summary": "summary of B2"},
    ]
    assert "text" not in str(state)


async def test_dead_end_reenables_parent_siblings() -> None:
    tree = tree_from_spec(SPEC)
    # Root prefers A; A's children are all bad; B is below tau_expand but is the best
    # remaining sibling; B1 is good.
    handler = by_title({"A": 0.9, "B": 0.4, "C": 0.3, "B1": 0.9})
    bs, _ = search(handler, Thresholds(tau_expand=0.5, tau_stop=0.7, tau_llm=0.0, beam_width=1))
    result = await bs.retrieve(tree, "q")
    expanded = [e.node_id for e in result.trace.expansions]
    assert expanded == ["Doc", "A", "B"]
    a = result.trace.expansions[1]
    assert a.dead_end
    root = result.trace.expansions[0]
    b_decision = next(c for c in root.children if c.child_id == "B")
    assert b_decision.expanded  # rescued sibling is visible in the trace
    assert result.node_ids == ["B1"]


async def test_no_rescue_at_root_gives_honest_empty_result() -> None:
    tree = tree_from_spec(SPEC)
    bs, _ = search(by_title({}))
    result = await bs.retrieve(tree, "q")
    assert result.nodes == [] and not result.no_answer
    assert result.trace.expansions[0].dead_end


async def test_budget_raises_instead_of_truncating() -> None:
    tree = tree_from_spec(SPEC)
    t = Thresholds(tau_expand=0.5, tau_stop=0.7, tau_llm=0.0, beam_width=2, max_expansions=2)
    bs, _ = search(by_title({"A": 0.9, "B": 0.9, "B2": 0.9}), t)
    with pytest.raises(TraversalBudgetExceeded):
        await bs.retrieve(tree, "q")


async def test_stop_here_emits_internal_node_and_keeps_descending() -> None:
    tree = tree_from_spec(SPEC)
    handler = by_title({"B": 0.9, "stop@B": 0.95, "B2": 0.8, "B2x": 0.6, "stop@B2": 0.1})
    bs, _ = search(handler)
    result = await bs.retrieve(tree, "q", max_nodes=5)
    assert result.node_ids[0] == "B"  # stop probability 0.95 outranks admission scores
    assert set(result.node_ids) == {"B", "B2x"}
    b = next(e for e in result.trace.expansions if e.node_id == "B")
    assert b.emitted and not b.dead_end


async def test_off_topic_at_root_returns_no_answer() -> None:
    tree = tree_from_spec(SPEC)
    bs, client = search(by_title({"A": 0.9, "off_topic": 0.97}))
    result = await bs.retrieve(tree, "q")
    assert result.no_answer and result.nodes == []
    assert len(client.calls) == 1
    assert result.trace.expansions[0].off_topic == pytest.approx(0.97)


async def test_beam_width_caps_level_across_parents() -> None:
    tree = tree_from_spec(SPEC)
    handler = by_title({"A": 0.9, "B": 0.8, "A1": 0.9, "A2": 0.8, "B1": 0.7, "B2": 0.6})
    bs, client = search(
        handler, Thresholds(tau_expand=0.5, tau_stop=0.7, tau_llm=0.0, beam_width=2)
    )
    result = await bs.retrieve(tree, "q", max_nodes=10)
    # Level 1 beam = [A, B]; level 2 candidates A1 .9, A2 .8, B1 .7, B2 .6 -> [A1, A2].
    assert result.node_ids == ["A1", "A2"]
    assert result.scores == [pytest.approx(0.9), pytest.approx(0.8)]
    assert len(client.calls) == 3


async def test_max_depth_emits_instead_of_expanding() -> None:
    tree = tree_from_spec(SPEC)
    handler = by_title({"B": 0.9, "B2": 0.9, "B2x": 0.9})
    t = Thresholds(tau_expand=0.5, tau_stop=0.7, tau_llm=0.0, beam_width=2, max_depth=2)
    bs, client = search(handler, t)
    result = await bs.retrieve(tree, "q")
    assert result.node_ids == ["B2"]
    assert [c["state"]["current_section"]["title"] for c in client.calls] == ["Doc", "B"]


async def test_trace_records_versioned_model_id() -> None:
    tree = tree_from_spec(SPEC)
    client = ScriptedJevClient(by_title({"A": 0.9}), model_id="jev-9.9.9")
    bs = BeamSearch(JevPolicy(client, NO_FALLBACK), NO_FALLBACK)
    result = await bs.retrieve(tree, "q")
    assert result.trace.model_ids == ["jev-9.9.9"]


def test_select_children_orders_and_caps() -> None:
    kids = [
        ChildDecision("a", 0, 0.6, 0.9, {}, False),
        ChildDecision("b", 0, 0.9, 0.9, {}, False),
        ChildDecision("c", 0, 0.3, 0.9, {}, False),
        ChildDecision("d", 0, 0.9, 0.9, {}, False),
    ]
    t = Thresholds(tau_expand=0.5, beam_width=2)
    assert select_children(kids, t) == [1, 3]  # ties keep document order
    assert select_children(kids, Thresholds(tau_expand=0.5, beam_width=5)) == [1, 3, 0]


def test_needs_escalation_only_on_decisive_children() -> None:
    t = Thresholds(tau_expand=0.5, tau_llm=0.4, beam_width=2)
    kids = [
        ChildDecision("a", 0, 0.9, 0.2, {}, False),  # selected, low confidence
        ChildDecision("b", 0, 0.1, 0.1, {}, False),  # irrelevant, uncertain: ignored
    ]
    assert needs_escalation(kids, t)
    kids[0].confidence = 0.8
    assert not needs_escalation(kids, t)
    # No child clears the floor: the best one decides.
    kids = [
        ChildDecision("a", 0, 0.3, 0.2, {}, False),
        ChildDecision("b", 0, 0.1, 0.9, {}, False),
    ]
    assert needs_escalation(kids, t)
    assert not needs_escalation(kids, t.without_fallback())


async def test_fallback_redecides_low_confidence_expansion() -> None:
    tree = tree_from_spec(SPEC)
    # Jev is unsure at the root (A .9 with confidence .2); the LLM says B instead.
    handler = by_title({"A": (0.9, 0.2), "B": (0.4, 0.3), "C": 0.1, "B1": 0.9})

    def llm(model: str, system: str, user: str, schema: Any) -> dict[str, Any]:
        assert '"current_section"' in user
        n = schema["properties"]["children"]["minItems"]
        levels = [0] * n
        if n == 3:
            levels = [0, 3, 0]
        return {"children": levels, "stop_here": 0.05}

    llm_client = ScriptedLlmClient(llm, model_id="claude-x")
    t = Thresholds(tau_expand=0.5, tau_stop=0.7, tau_llm=0.4, beam_width=1)
    policy = JevPolicy(ScriptedJevClient(handler), t, LlmFallback(llm_client, "claude-x"))
    result = await BeamSearch(policy, t).retrieve(tree, "q")
    root = result.trace.expansions[0]
    assert root.escalated_to_llm
    assert root.model_id == "jev-1.13.0+claude-x"
    assert root.jev_children is not None and root.jev_children[0].normalized == pytest.approx(
        0.9
    )
    assert [c.normalized for c in root.children] == [0.0, 1.0, 0.0]
    assert result.node_ids == ["B1"]
    assert result.trace.escalation_rate == pytest.approx(1 / 2)
    assert len(llm_client.calls) == 1


async def test_fallback_is_capped_per_query() -> None:
    tree = tree_from_spec(SPEC)
    handler = by_title({"B": (0.9, 0.1), "B2": (0.9, 0.1), "B2x": (0.9, 0.1)})
    llm_client = ScriptedLlmClient(
        lambda m, s, u, sc: {
            "children": [3] * sc["properties"]["children"]["minItems"],
            "stop_here": 0.0,
        }
    )
    t = Thresholds(tau_expand=0.5, tau_stop=0.7, tau_llm=0.4, beam_width=3)
    policy = JevPolicy(
        ScriptedJevClient(handler), t, LlmFallback(llm_client, "m", max_escalations_per_query=1)
    )
    await BeamSearch(policy, t).retrieve(tree, "q1")
    assert len(llm_client.calls) == 1
    await BeamSearch(policy, t).retrieve(tree, "q2")
    assert len(llm_client.calls) == 2  # the counter resets per query


def test_fallback_threshold_without_fallback_object_is_rejected() -> None:
    with pytest.raises(ValueError):
        JevPolicy(ScriptedJevClient(by_title({})), Thresholds(tau_llm=0.4))


async def test_llm_policy_runs_in_the_same_beam() -> None:
    tree = tree_from_spec(SPEC)

    def llm(model: str, system: str, user: str, schema: Any) -> dict[str, Any]:
        import json

        state = json.loads(user.split("State:\n", 1)[1].rsplit("\n\nReturn one", 1)[0])
        levels = [3 if c["title"] in ("B", "B1") else 0 for c in state["children"]]
        return {"children": levels, "stop_here": 0.0}

    llm_client = ScriptedLlmClient(llm, model_id="claude-x")
    policy = LlmPolicy(llm_client, "claude-x", NO_FALLBACK)
    result = await BeamSearch(policy, NO_FALLBACK).retrieve(tree, "q")
    assert result.node_ids == ["B1"]
    assert result.trace.model_ids == ["claude-x"]
    assert all(c.confidence == 1.0 for e in result.trace.expansions for c in e.children)
    assert "State:" in llm_client.calls[0]["user"]
