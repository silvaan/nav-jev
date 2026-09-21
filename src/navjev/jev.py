"""Thin async wrapper over the TypeSafe System One API.

One rule governs this module: a request carries one state and every question that can be
answered from it. Looping over children with one call each is a bug. See CLAUDE.md.

Reference shape, from TypeSafe's API:

    POST https://api.typesafe.ai/v1/systemone
    {"model": "jev-latest", "state": {...}, "questions": {"<id>": {...}}}

    -> {"model": "jev-1.13.0", "answers": {...}, "usage": {"input_tokens": n, ...}}

The wire format is produced here from the project's own question dataclasses and sent
through the official `typesafe-sdk` client, whose retry policy already covers 429 and
529 with exponential backoff. Responses are consumed from the raw JSON body so that a
recorded fixture and a live answer go through the same parsing path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

State = str | Mapping[str, Any] | Sequence[str]

# Documented budget: state plus all questions share ~64k tokens, and state plus the
# longest single question must fit in ~32k. The client checks before sending so an
# oversized expansion fails locally instead of as a 422.
MAX_TOTAL_TOKENS = 64_000
MAX_STATE_PLUS_QUESTION_TOKENS = 32_000
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# There is no public tokenizer for Jev. Four characters per token is the usual rough
# estimate for English prose and JSON; the local check is a guard against gross
# overruns, not an exact count, and the API remains the authority.
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Noul:
    """Yes/no. Phrase it so a high probability means yes; a Noul whose `true` means "no"
    degrades the answer. There is no confidence field: the probability is the signal,
    and 0.5 means the model cannot tell, not "medium"."""

    instructions: str
    criteria: Mapping[str, str] | None = None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            out["criteria"] = dict(self.criteria)
        return out


@dataclass(frozen=True)
class Choice:
    """Pick one of up to 255 unordered options. Always include an `other` escape when the
    option set may not cover every input, because the model must pick something."""

    instructions: str
    criteria: Mapping[str, str | None]

    def to_wire(self) -> dict[str, Any]:
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise JevError(
                f"Choice has {len(self.criteria)} options; the API accepts at most "
                f"{MAX_CHOICE_OPTIONS}"
            )
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True)
class Score:
    """A position on a scale. `criteria` is ordered low to high, 2 to 10 levels, and each
    level describes a *situation*, never a degree. A level may be a mapping with `what`
    and `examples`; examples that resemble real inputs measurably sharpen confidence.

    The returned `score` is the probability-weighted mean of the level indices, so divide
    by `len(criteria) - 1` before comparing against anything.
    """

    instructions: str
    criteria: Sequence[str | Mapping[str, Any]]

    @property
    def levels(self) -> int:
        return len(self.criteria)

    def to_wire(self) -> dict[str, Any]:
        n = len(self.criteria)
        if not MIN_SCORE_LEVELS <= n <= MAX_SCORE_LEVELS:
            raise JevError(
                f"Score has {n} levels; the API accepts {MIN_SCORE_LEVELS} to "
                f"{MAX_SCORE_LEVELS}"
            )
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": [c if isinstance(c, str) else dict(c) for c in self.criteria],
        }


Question = Noul | Choice | Score


@dataclass
class Answer:
    question_id: str
    type: str
    value: float | str
    probabilities: Mapping[str, float] | None
    confidence: float | None
    legend: Mapping[str, str] | None = None

    def normalized(self, levels: int) -> float:
        """Score answers only: value divided by the top level index."""
        if self.type != "score":
            raise JevError(f"normalized() is only defined for score answers, not {self.type}")
        if levels < MIN_SCORE_LEVELS:
            raise JevError(f"a score needs at least {MIN_SCORE_LEVELS} levels, got {levels}")
        return float(self.value) / (levels - 1)

    @property
    def probability(self) -> float:
        """Noul answers only: the probability of yes."""
        if self.type != "noul":
            raise JevError(f"probability is only defined for noul answers, not {self.type}")
        return float(self.value)

    @classmethod
    def from_wire(cls, question_id: str, raw: Mapping[str, Any]) -> Answer:
        kind = raw["type"]
        if kind == "noul":
            return cls(question_id, "noul", float(raw["noul"]), None, None)
        if kind == "choice":
            return cls(
                question_id,
                "choice",
                str(raw["choice"]),
                {str(k): float(v) for k, v in raw["probabilities"].items()},
                float(raw["confidence"]),
            )
        if kind == "score":
            legend = raw.get("legend")
            return cls(
                question_id,
                "score",
                float(raw["score"]),
                {str(k): float(v) for k, v in raw["probabilities"].items()},
                float(raw["confidence"]),
                {str(k): str(v) for k, v in legend.items()} if legend else None,
            )
        raise JevError(f"unknown answer type {kind!r} for question {question_id!r}")


@dataclass
class JevResponse:
    model: str
    """Versioned ID that answered. Log it; pin it once thresholds are tuned."""
    answers: dict[str, Answer]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    request_id: str | None = None

    def cost_usd(self, rate_per_million: float) -> float:
        """Rate comes from the config, never from a constant in the code."""
        return self.input_tokens / 1_000_000 * rate_per_million

    @classmethod
    def from_wire(cls, raw: Mapping[str, Any], latency_ms: float) -> JevResponse:
        usage = raw.get("usage") or {}
        answers = {qid: Answer.from_wire(qid, a) for qid, a in raw["answers"].items()}
        return cls(
            model=str(raw["model"]),
            answers=answers,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            latency_ms=latency_ms,
            request_id=raw.get("request_id"),
        )


class JevError(RuntimeError):
    pass


class JevBudgetExceeded(JevError):
    pass


class JevFixtureMiss(JevError):
    """A recorded client was asked something the fixture does not hold."""


def estimate_tokens(obj: Any) -> int:
    """Rough token count of a JSON-serialisable value. See `_CHARS_PER_TOKEN`."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return max(1, len(text) // _CHARS_PER_TOKEN)


def request_key(state: State, questions: Mapping[str, Any]) -> str:
    """Stable hash of a request body, used to key recorded fixtures."""
    wire = {q: (v.to_wire() if hasattr(v, "to_wire") else v) for q, v in questions.items()}
    body = json.dumps(
        {"state": _plain(state), "questions": wire}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _plain(state: State) -> Any:
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping):
        return dict(state)
    return list(state)


@dataclass
class JevUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    spent_usd: float = 0.0
    model_ids: list[str] = field(default_factory=list)


class JevClient:
    """Async client with retry, concurrency limiting, and spend accounting.

    Retries 429 and 529 with exponential backoff. Does not retry 401 or 422; a 422 names
    the offending field and is a bug in the caller, not a transient failure.

    `max_spend_usd` defaults to zero, which blocks every paid call. Live tests and
    benchmarks must raise it explicitly in their config, so an accidental run over a full
    split cannot silently spend money.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jev-latest",
        max_concurrency: int = 8,
        max_spend_usd: float = 0.0,
        rate_per_million: float = 0.042,
        timeout_s: float = 30.0,
        base_url: str | None = None,
        record_to: str | Path | None = None,
        max_retries: int = 5,
    ) -> None:
        self.model = model
        self.rate_per_million = rate_per_million
        self.max_spend_usd = max_spend_usd
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._api_key = api_key
        self._base_url = base_url
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._sdk: Any = None
        self._usage = JevUsage()
        self._record_to = Path(record_to) if record_to else None
        self._lock = asyncio.Lock()

    # -- accounting -------------------------------------------------------------------

    @property
    def spent_usd(self) -> float:
        return self._usage.spent_usd

    @property
    def usage(self) -> JevUsage:
        return self._usage

    def _check_budget(self) -> None:
        if self.max_spend_usd <= 0.0:
            raise JevBudgetExceeded(
                "max_spend_usd is zero: every paid call is blocked. Raise it explicitly "
                "in the config to allow live requests."
            )
        if self._usage.spent_usd >= self.max_spend_usd:
            raise JevBudgetExceeded(
                f"spent ${self._usage.spent_usd:.4f} of a ${self.max_spend_usd:.4f} cap"
            )

    def _account(self, response: JevResponse) -> None:
        self._usage.requests += 1
        self._usage.input_tokens += response.input_tokens
        self._usage.output_tokens += response.output_tokens
        self._usage.spent_usd += response.cost_usd(self.rate_per_million)
        if response.model not in self._usage.model_ids:
            self._usage.model_ids.append(response.model)

    # -- validation -------------------------------------------------------------------

    @staticmethod
    def _validate(state: State, questions: Mapping[str, Question]) -> dict[str, Any]:
        if not questions:
            raise JevError("a request needs at least one question")
        wire = {qid: q.to_wire() for qid, q in questions.items()}
        state_tokens = estimate_tokens(_plain(state))
        longest = max(estimate_tokens(w) for w in wire.values())
        total = state_tokens + sum(estimate_tokens(w) for w in wire.values())
        if state_tokens + longest > MAX_STATE_PLUS_QUESTION_TOKENS:
            raise JevError(
                f"state plus longest question is ~{state_tokens + longest} tokens, over "
                f"the {MAX_STATE_PLUS_QUESTION_TOKENS} limit"
            )
        if total > MAX_TOTAL_TOKENS:
            raise JevError(
                f"state plus all questions is ~{total} tokens, over the "
                f"{MAX_TOTAL_TOKENS} limit"
            )
        return wire

    # -- transport --------------------------------------------------------------------

    def _sdk_client(self) -> Any:
        if self._sdk is None:
            from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

            api_key = self._api_key or os.environ.get("TYPESAFE_API_KEY")
            if not api_key:
                raise JevError("TYPESAFE_API_KEY is not set and no api_key was given")
            # 401 and 422 are not in the SDK's retried statuses; 429 and 5xx (529
            # included) are. Backoff is exponential with jitter.
            policy = RetryPolicy(
                max_retries=self.max_retries, backoff_initial=0.5, backoff_max=8.0
            )
            self._sdk = AsyncTypeSafeClient(
                api_key=api_key,
                model=self.model,
                retry=policy,
                timeout=self.timeout_s,
                base_url=self._base_url,
            )
        return self._sdk

    async def _send(self, state: State, wire: Mapping[str, Any]) -> dict[str, Any]:
        """One HTTP round trip. Returns the raw JSON body. Subclasses replace this."""
        from typesafe_sdk import TypeSafeAPIError

        client = self._sdk_client()
        try:
            result = await client.system_one(_plain(state), dict(wire))
        except TypeSafeAPIError as error:  # pragma: no cover - live only
            raise JevError(f"Jev request failed: {error}") from error
        body: dict[str, Any] = result.raw_http_response.json()
        body["request_id"] = result.request_id
        return body

    async def ask(self, state: State, questions: Mapping[str, Question]) -> JevResponse:
        """One request, every question answered in parallel against the same state."""
        wire = self._validate(state, questions)
        self._check_budget()
        async with self._semaphore:
            started = time.perf_counter()
            body = await self._send(state, wire)
            latency_ms = (time.perf_counter() - started) * 1000.0
        response = JevResponse.from_wire(body, latency_ms)
        async with self._lock:
            self._account(response)
            if self._record_to is not None:
                self._record(state, wire, body)
        return response

    async def ask_many(
        self, batches: Sequence[tuple[State, Mapping[str, Question]]]
    ) -> list[JevResponse]:
        """Concurrent independent requests, for expanding a whole beam level at once."""
        return list(await asyncio.gather(*(self.ask(s, q) for s, q in batches)))

    async def aclose(self) -> None:
        if self._sdk is not None:
            await self._sdk.aclose()
            self._sdk = None

    # -- recording --------------------------------------------------------------------

    def _record(self, state: State, wire: Mapping[str, Any], body: Mapping[str, Any]) -> None:
        assert self._record_to is not None
        entry = {
            "key": request_key(state, wire),
            "request": {"state": _plain(state), "questions": dict(wire)},
            "response": dict(body),
        }
        self._record_to.parent.mkdir(parents=True, exist_ok=True)
        with self._record_to.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


class RecordedJevClient(JevClient):
    """Replays responses from a fixture file so the unit suite needs no API key.

    Keyed by a hash of state plus questions. A miss raises rather than falling through to
    the network, which keeps an offline test from quietly becoming a paid one.

    The fixture is JSON Lines, one `{"key", "request", "response"}` object per line, which
    is exactly what `JevClient(record_to=...)` writes during a live session.
    """

    def __init__(
        self,
        fixture_path: str | Path | None = None,
        model: str = "jev-latest",
        max_concurrency: int = 8,
        rate_per_million: float = 0.042,
    ) -> None:
        # Replay is free, so the spend cap is lifted: the point of the cap is to stop
        # paid calls, and this client cannot make one.
        super().__init__(
            api_key="recorded",
            model=model,
            max_concurrency=max_concurrency,
            max_spend_usd=float("inf"),
            rate_per_million=rate_per_million,
        )
        self.fixture_path = Path(fixture_path) if fixture_path else None
        self._responses: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []
        if self.fixture_path is not None:
            self._load(self.fixture_path)

    def _load(self, path: Path) -> None:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._responses[entry["key"]] = entry["response"]

    def add(
        self, state: State, questions: Mapping[str, Any], response: Mapping[str, Any]
    ) -> None:
        """Register a response for a request, for tests that build fixtures in code."""
        self._responses[request_key(state, questions)] = dict(response)

    async def _send(self, state: State, wire: Mapping[str, Any]) -> dict[str, Any]:
        key = request_key(state, wire)
        self.calls.append({"key": key, "state": _plain(state), "questions": dict(wire)})
        try:
            return dict(self._responses[key])
        except KeyError:
            raise JevFixtureMiss(
                f"no recorded response for request {key[:12]}… "
                f"({len(wire)} questions). Record it with JevClient(record_to=...)."
            ) from None
