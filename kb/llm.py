"""LLM client for OpenCode: Claude models over the Anthropic Messages API (official
SDK), other models over the OpenAI-compatible endpoint. JSON output, retries,
usage/cost accounting and Langfuse generation logging."""
from __future__ import annotations

import contextvars
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import lru_cache

import anthropic
import httpx

from . import config

_lf = None
if config.LANGFUSE_ENABLED:
    try:
        from langfuse import get_client

        _lf = get_client()
    except Exception:  # tracing must never break answering
        _lf = None


def langfuse():
    return _lf


class Usage:
    """Token / cost / call counters for one request (thread-safe)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0

    def add(self, p: int, c: int, cost: float):
        with self._lock:
            self.calls += 1
            self.prompt_tokens += p
            self.completion_tokens += c
            self.cost += cost

    def to_dict(self) -> dict:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost, 6),
        }


class _Retryable(Exception):
    pass


class _Fatal(Exception):
    """A provider error that retrying cannot fix (spent budget / quota): fail fast so callers
    and proxies (e.g. Cloudflare's 100 s limit) get an error in seconds, not minutes."""


def _is_budget_error(text: str) -> bool:
    t = text.lower()
    return "budget" in t or "quota" in t or "insufficient" in t or "credit" in t


_reasoning_ok = True
_http = httpx.Client(timeout=httpx.Timeout(180.0, connect=15.0), limits=httpx.Limits(max_connections=32))


@lru_cache(maxsize=1)
def _claude() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=config.OPENCODE_API_KEY, base_url=config.ANTHROPIC_BASE_URL,
                               timeout=180.0, max_retries=0)  # retries are handled in chat(); the SDK would honour long retry-after waits


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


# how each Claude model turns thinking off (Sonnet 5.5 rejects {"type": "disabled"})
_NO_THINKING = {"claude-sonnet-5-5": {"type": "between_tools"}}


def _call_claude(model: str, system: str, user: str, max_tokens: int, json_mode: bool) -> tuple[str, bool, int, int]:
    """Returns (text, truncated, input_tokens, output_tokens). Thinking is disabled:
    every step here is extraction / classification / grounded writing."""
    if json_mode:
        system += "\n\nRespond with a single JSON object only, no prose and no code fences."
    try:
        r = _claude().messages.create(
            model=model,
            max_tokens=max_tokens,
            thinking=_NO_THINKING.get(model, {"type": "disabled"}),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.RateLimitError as e:
        raise (_Fatal if _is_budget_error(str(e)) else _Retryable)(str(e)) from e
    except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
        raise _Retryable(str(e)) from e
    u = r.usage
    p = (u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, r.stop_reason == "max_tokens", p, u.output_tokens or 0


def _call_openai(model: str, system: str, user: str, max_tokens: int, json_mode: bool, reasoning: str) -> tuple[str, bool, int, int]:
    global _reasoning_ok
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if reasoning and _reasoning_ok:
        payload["reasoning_effort"] = reasoning
    headers = {"Authorization": f"Bearer {config.OPENCODE_API_KEY}", "Content-Type": "application/json"}
    try:
        r = _http.post(config.OPENCODE_BASE_URL, headers=headers, json=payload)
        if r.status_code == 400 and "reasoning" in r.text.lower() and "reasoning_effort" in payload:
            _reasoning_ok = False  # provider no longer accepts this effort value
            payload.pop("reasoning_effort")
            r = _http.post(config.OPENCODE_BASE_URL, headers=headers, json=payload)
    except httpx.HTTPError as e:
        raise _Retryable(str(e)) from e
    if r.status_code == 429 and _is_budget_error(r.text):
        raise _Fatal(f"{r.status_code}: {r.text[:200]}")
    if r.status_code == 429 or r.status_code >= 500:
        raise _Retryable(f"{r.status_code}: {r.text[:200]}")
    r.raise_for_status()
    data = r.json()
    u = data.get("usage") or {}
    choice = data["choices"][0]
    return (choice["message"].get("content") or "", choice.get("finish_reason") == "length",
            int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))


def chat(
    system: str,
    user: str,
    *,
    model: str | None = None,
    json_mode: bool = True,
    max_tokens: int = 4000,
    usage: Usage | None = None,
    name: str = "llm",
    retries: int = 3,
    reasoning: str | None = None,
):
    """Single completion. Returns parsed dict when json_mode else str.

    Claude models (`claude-*`) go through the Anthropic Messages API with thinking
    disabled; other models through the OpenAI-compatible endpoint, where `reasoning`
    (default config.REASONING_EFFORT) is sent as reasoning_effort and dropped for the
    rest of the process if the provider rejects it."""
    if not config.OPENCODE_API_KEY:
        raise RuntimeError("OPENCODE_API_KEY is not set (put it in .env)")
    model = model or config.FAST_MODEL
    reasoning = reasoning if reasoning is not None else config.REASONING_EFFORT
    price_in, price_out = config.price(model)

    ctx = (
        _lf.start_as_current_observation(name=name, as_type="generation", model=model, input={"system": system[:2000], "user": user[:8000]})
        if _lf
        else nullcontext()
    )
    last: Exception | None = None
    with ctx as gen:
        for attempt in range(retries):
            try:
                if model.startswith("claude"):
                    text, truncated, p, c = _call_claude(model, system, user, max_tokens, json_mode)
                else:
                    text, truncated, p, c = _call_openai(model, system, user, max_tokens, json_mode, reasoning)
                cost = (p * price_in + c * price_out) / 1e6
                if usage:
                    usage.add(p, c, cost)  # failed attempts are still billed
                if not text.strip() or (json_mode and truncated):
                    if truncated:
                        max_tokens *= 2  # truncated: retry with room
                    raise ValueError(f"empty/truncated content (truncated={truncated})")
                out = _parse_json(text) if json_mode else text
                if gen is not None:
                    gen.update(output=text[:8000], usage_details={"input": p, "output": c},
                               cost_details={"input": p * price_in / 1e6, "output": c * price_out / 1e6})
                return out
            except (_Retryable, _Fatal, httpx.HTTPError, anthropic.APIStatusError, KeyError, ValueError, json.JSONDecodeError) as e:
                last = e
                if isinstance(e, (_Fatal, anthropic.BadRequestError)):
                    break  # malformed request: retrying will not help
                time.sleep(1.5 * (attempt + 1))
        if gen is not None:
            gen.update(level="ERROR", status_message=str(last)[:500])
    raise RuntimeError(f"LLM call '{name}' failed: {last}")


def chat_many(jobs: list[dict], concurrency: int | None = None) -> list:
    """Run several chat() calls in parallel; each job is kwargs for chat().
    Failed jobs return None instead of raising."""

    def run(kw):
        try:
            return chat(**kw)
        except Exception:
            return None

    if not jobs:
        return []
    # copy the caller's context per job so Langfuse nests the calls under the current span
    with ThreadPoolExecutor(max_workers=concurrency or config.LLM_CONCURRENCY) as ex:
        futs = [ex.submit(contextvars.copy_context().run, run, kw) for kw in jobs]
        return [f.result() for f in futs]
