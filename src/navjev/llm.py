"""The one place that talks to a text-generating LLM.

Two callers share it: node summarization at index time and the optional LLM fallback
in the traversal. Each call
returns JSON that matches a schema the caller supplies, so no caller parses free text.

Two providers, chosen by model name: `claude-*` goes to the Anthropic Messages API with
`output_config.format`, and `gpt-*` / `o*` go to the OpenAI chat completions API with a
strict `json_schema` response format. Both return the same `LlmCompletion`, and the
usage fields both report are what the cost accounting multiplies by the config rates.

Pricing is never hardcoded. The caller passes a `rate_per_million` pair from the config;
a model with no rate in the config is accounted at zero and flagged in `usage`, which the
manifest reports rather than hides.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class LlmError(RuntimeError):
    pass


class LlmBudgetExceeded(LlmError):
    pass


class LlmFixtureMiss(LlmError):
    pass


@dataclass(frozen=True)
class LlmRate:
    input_per_million: float
    output_per_million: float


@dataclass
class LlmCompletion:
    model: str
    data: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


@dataclass
class LlmUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    spent_usd: float = 0.0
    unpriced_models: list[str] = field(default_factory=list)
    """Models called without a rate in the config. Their cost is unknown, not zero."""


def request_key(model: str, system: str, user: str, schema: Mapping[str, Any]) -> str:
    body = json.dumps(
        {"model": model, "system": system, "user": user, "schema": schema}, sort_keys=True
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class LlmClient:
    """Async client over the OpenAI or Anthropic API with JSON-schema output.

    `max_spend_usd` works as in `JevClient`: `None` is uncapped, `0.0` blocks every call.
    """

    def __init__(
        self,
        rates: Mapping[str, LlmRate] | None = None,
        max_spend_usd: float | None = None,
        max_concurrency: int = 4,
        record_to: str | Path | None = None,
        api_key: str | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        self.rates = dict(rates or {})
        self.max_spend_usd = max_spend_usd
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()
        self._sdk: Any = None
        self._openai: Any = None
        self._api_key = api_key
        self._openai_key = openai_api_key
        self._usage = LlmUsage()
        self._record_to = Path(record_to) if record_to else None

    @property
    def usage(self) -> LlmUsage:
        return self._usage

    @property
    def spent_usd(self) -> float:
        return self._usage.spent_usd

    def cost_of(self, model: str, input_tokens: int, output_tokens: int) -> float:
        rate = self.rates.get(model)
        if rate is None:
            if model not in self._usage.unpriced_models:
                self._usage.unpriced_models.append(model)
            return 0.0
        return (
            input_tokens / 1_000_000 * rate.input_per_million
            + output_tokens / 1_000_000 * rate.output_per_million
        )

    def _check_budget(self) -> None:
        if self.max_spend_usd is None:
            return
        if self.max_spend_usd <= 0.0:
            raise LlmBudgetExceeded("max_spend_usd is zero: every LLM call is blocked")
        if self._usage.spent_usd >= self.max_spend_usd:
            raise LlmBudgetExceeded(
                f"spent ${self._usage.spent_usd:.4f} of a ${self.max_spend_usd:.4f} cap"
            )

    def _sdk_client(self) -> Any:
        if self._sdk is None:
            import anthropic

            self._sdk = anthropic.AsyncAnthropic(api_key=self._api_key, max_retries=4)
        return self._sdk

    def _openai_client(self) -> Any:
        if self._openai is None:
            import httpx

            key = self._openai_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise LlmError("OPENAI_API_KEY is not set")
            self._openai = httpx.AsyncClient(
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                headers={"Authorization": f"Bearer {key}"},
                timeout=120.0,
            )
        return self._openai

    @staticmethod
    def provider_of(model: str) -> str:
        if model.startswith("claude"):
            return "anthropic"
        if model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
            return "openai"
        raise LlmError(f"no provider known for model {model!r}")

    async def _send(
        self,
        model: str,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        """One round trip. Returns `{"model", "text", "input_tokens", "output_tokens"}`."""
        if self.provider_of(model) == "openai":
            return await self._send_openai(model, system, user, schema, max_tokens)
        return await self._send_anthropic(model, system, user, schema, max_tokens)

    async def _send_openai(
        self,
        model: str,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        import httpx

        client = self._openai_client()
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": dict(schema)},
            },
        }
        delay = 1.0
        for attempt in range(6):
            try:
                response = await client.post("/chat/completions", json=body)
            except httpx.HTTPError as error:
                if attempt == 5:
                    raise LlmError(f"{model}: connection error: {error}") from error
                await asyncio.sleep(delay)
                delay = min(delay * 2, 16.0)
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                retry_after = response.headers.get("retry-after")
                await asyncio.sleep(float(retry_after) if retry_after else delay)
                delay = min(delay * 2, 16.0)
                continue
            if response.status_code >= 400:
                raise LlmError(f"{model}: {response.status_code} {response.text[:300]}")
            break
        payload = response.json()
        choice = payload["choices"][0]
        message = choice["message"]
        if message.get("refusal"):
            raise LlmError(f"{model} refused the request: {message['refusal']}")
        if choice.get("finish_reason") == "length":
            raise LlmError(f"{model} hit max_completion_tokens={max_tokens} before finishing")
        usage = payload.get("usage") or {}
        return {
            "model": str(payload.get("model", model)),
            "text": str(message.get("content") or ""),
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        }

    async def _send_anthropic(
        self,
        model: str,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        import anthropic

        client = self._sdk_client()
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": dict(schema)}},
            )
        except anthropic.APIStatusError as error:  # pragma: no cover - live only
            raise LlmError(f"{model}: {error.status_code} {error.message}") from error
        except anthropic.APIConnectionError as error:  # pragma: no cover - live only
            raise LlmError(f"{model}: connection error: {error}") from error
        if response.stop_reason == "refusal":  # pragma: no cover - live only
            raise LlmError(f"{model} refused the request")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return {
            "model": response.model,
            "text": text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    async def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        max_tokens: int = 2048,
    ) -> LlmCompletion:
        self._check_budget()
        async with self._semaphore:
            started = time.perf_counter()
            raw = await self._send(model, system, user, schema, max_tokens)
            latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            data = json.loads(raw["text"])
        except json.JSONDecodeError as error:
            raise LlmError(
                f"{model} returned non-JSON output: {raw['text'][:200]!r}"
            ) from error
        if not isinstance(data, dict):
            raise LlmError(f"{model} returned a JSON {type(data).__name__}, expected an object")
        # Priced by the requested ID first, then by the resolved one, so a config that
        # names an alias still prices what it asked for.
        priced_model = model if model in self.rates else str(raw["model"])
        cost = self.cost_of(priced_model, int(raw["input_tokens"]), int(raw["output_tokens"]))
        completion = LlmCompletion(
            model=str(raw["model"]),
            data=data,
            input_tokens=int(raw["input_tokens"]),
            output_tokens=int(raw["output_tokens"]),
            latency_ms=latency_ms,
            cost_usd=cost,
        )
        async with self._lock:
            self._usage.requests += 1
            self._usage.input_tokens += completion.input_tokens
            self._usage.output_tokens += completion.output_tokens
            self._usage.spent_usd += cost
            if self._record_to is not None:
                self._record(model, system, user, schema, raw)
        return completion

    def _record(
        self,
        model: str,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        raw: Mapping[str, Any],
    ) -> None:
        assert self._record_to is not None
        entry = {
            "key": request_key(model, system, user, schema),
            "request": {"model": model, "system": system, "user": user, "schema": dict(schema)},
            "response": dict(raw),
        }
        self._record_to.parent.mkdir(parents=True, exist_ok=True)
        with self._record_to.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def aclose(self) -> None:
        if self._sdk is not None:
            await self._sdk.close()
            self._sdk = None
        if self._openai is not None:
            await self._openai.aclose()
            self._openai = None


class RecordedLlmClient(LlmClient):
    """Replays completions from a JSON Lines fixture. A miss raises."""

    def __init__(
        self,
        fixture_path: str | Path | None = None,
        rates: Mapping[str, LlmRate] | None = None,
    ) -> None:
        super().__init__(rates=rates, max_spend_usd=float("inf"), api_key="recorded")
        self._responses: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []
        if fixture_path is not None:
            with Path(fixture_path).open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        entry = json.loads(line)
                        self._responses[entry["key"]] = entry["response"]

    def add(
        self,
        model: str,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        data: Mapping[str, Any],
        input_tokens: int = 100,
        output_tokens: int = 20,
    ) -> None:
        self._responses[request_key(model, system, user, schema)] = {
            "model": model,
            "text": json.dumps(data),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    async def _send(
        self,
        model: str,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        key = request_key(model, system, user, schema)
        self.calls.append({"key": key, "model": model, "system": system, "user": user})
        try:
            return dict(self._responses[key])
        except KeyError:
            raise LlmFixtureMiss(
                f"no recorded completion for {model} request {key[:12]}…"
            ) from None


class ScriptedLlmClient(LlmClient):
    """Answers from a callable, for tests that need behaviour rather than a fixture."""

    def __init__(
        self,
        handler: Any,
        model_id: str = "scripted-llm",
        rates: Mapping[str, LlmRate] | None = None,
    ) -> None:
        super().__init__(rates=rates, max_spend_usd=float("inf"), api_key="scripted")
        self._handler = handler
        self._model_id = model_id
        self.calls: list[dict[str, Any]] = []

    async def _send(
        self,
        model: str,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        self.calls.append({"model": model, "system": system, "user": user, "schema": schema})
        data = self._handler(model, system, user, schema)
        if asyncio.iscoroutine(data):
            data = await data
        return {
            "model": self._model_id,
            "text": json.dumps(data),
            "input_tokens": max(1, len(system + user) // 4),
            "output_tokens": max(1, len(json.dumps(data)) // 4),
        }
