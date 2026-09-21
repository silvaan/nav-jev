from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from navjev.llm import LlmBudgetExceeded, LlmClient, LlmError, LlmRate


def test_provider_dispatch_by_model_name() -> None:
    assert LlmClient.provider_of("claude-sonnet-5") == "anthropic"
    assert LlmClient.provider_of("gpt-4.1-mini") == "openai"
    assert LlmClient.provider_of("o4-mini") == "openai"
    with pytest.raises(LlmError):
        LlmClient.provider_of("llama-3")


def _openai_client(handler: Any, **kw: Any) -> LlmClient:
    client = LlmClient(
        rates={"gpt-4.1-mini": LlmRate(0.40, 1.60)}, max_spend_usd=1.0, openai_api_key="k", **kw
    )
    client._openai = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        headers={"Authorization": "Bearer k"},
        transport=httpx.MockTransport(handler),
    )
    return client


async def test_openai_backend_sends_strict_schema_and_prices_usage() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(
            200,
            json={
                "model": "gpt-4.1-mini-2025-04-14",
                "choices": [
                    {"message": {"content": '{"summary": "x"}'}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            },
        )

    client = _openai_client(handler)
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    }
    out = await client.complete_json("gpt-4.1-mini", "sys", "usr", schema, max_tokens=64)
    assert out.data == {"summary": "x"} and out.model == "gpt-4.1-mini-2025-04-14"
    assert out.cost_usd == pytest.approx(1000 / 1e6 * 0.40 + 100 / 1e6 * 1.60)
    body = seen[0]
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert body["max_completion_tokens"] == 64
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert client.usage.unpriced_models == []


async def test_openai_backend_retries_429_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429, headers={"retry-after": "0"}, json={"error": "slow down"}
            )
        return httpx.Response(
            200,
            json={
                "model": "gpt-4.1-mini",
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = _openai_client(handler)
    out = await client.complete_json("gpt-4.1-mini", "s", "u", {"type": "object"}, 8)
    assert calls == 2 and out.data == {}


async def test_openai_backend_surfaces_refusal_truncation_and_errors() -> None:
    def refusal(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [
                    {"message": {"content": None, "refusal": "no"}, "finish_reason": "stop"}
                ],
                "usage": {},
            },
        )

    with pytest.raises(LlmError, match="refused"):
        await _openai_client(refusal).complete_json(
            "gpt-4.1-mini", "s", "u", {"type": "object"}, 8
        )

    def truncated(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"content": "{"}, "finish_reason": "length"}],
                "usage": {},
            },
        )

    with pytest.raises(LlmError, match="max_completion_tokens"):
        await _openai_client(truncated).complete_json(
            "gpt-4.1-mini", "s", "u", {"type": "object"}, 8
        )

    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "invalid schema"}})

    with pytest.raises(LlmError, match="400"):
        await _openai_client(bad).complete_json("gpt-4.1-mini", "s", "u", {"type": "object"}, 8)


async def test_zero_cap_blocks_before_any_request() -> None:
    client = LlmClient(max_spend_usd=0.0, openai_api_key="k")
    with pytest.raises(LlmBudgetExceeded):
        await client.complete_json("gpt-4.1-mini", "s", "u", {"type": "object"}, 8)
