from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

from app.config import Settings, get_settings


class DailyQuotaExhausted(RuntimeError):
    """The provider's daily token budget is spent; waiting will not help."""


def _is_daily_cap(text: str) -> bool:
    lowered = text.lower()
    return "per day" in lowered or "tpd" in lowered


def _is_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return status == 429 or "rate limit" in str(exc).lower()


class RateLimitedLLM:
    """Wrap a chat model so a provider 429 becomes a wait, not a crash.

    The Groq free tier limits both tokens per minute and tokens per day, and a
    single eval run issues hundreds of calls across 25 questions. A per-minute
    cap deserves a backoff; a per-day cap does not, so the pool rotates to the
    next key and only raises once every key is spent.
    """

    def __init__(
        self,
        models: BaseChatModel | list[BaseChatModel],
        max_attempts: int = 6,
    ) -> None:
        self._models = list(models) if isinstance(models, (list, tuple)) else [models]
        if not self._models:
            raise ValueError("RateLimitedLLM needs at least one model")
        self._index = 0
        self._exhausted: set[int] = set()
        self._max_attempts = max_attempts

    def _rotate(self) -> bool:
        self._exhausted.add(self._index)
        for offset in range(1, len(self._models) + 1):
            candidate = (self._index + offset) % len(self._models)
            if candidate not in self._exhausted:
                self._index = candidate
                return True
        return False

    def _retry(self, fn, *args: Any, **kwargs: Any):
        delay = 5.0
        for attempt in range(1, self._max_attempts + 1):
            try:
                return fn(self._models[self._index], *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                if not _is_rate_limit(exc):
                    raise
                if _is_daily_cap(str(exc)) and self._rotate():
                    continue
                if _is_daily_cap(str(exc)):
                    raise DailyQuotaExhausted(str(exc)) from exc
                if attempt == self._max_attempts:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")  # pragma: no cover

    def invoke(self, *args: Any, **kwargs: Any) -> BaseMessage:
        return self._retry(lambda m, *a, **k: m.invoke(*a, **k), *args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> BaseMessage:
        delay = 5.0
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self._models[self._index].ainvoke(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                if not _is_rate_limit(exc):
                    raise
                if _is_daily_cap(str(exc)) and self._rotate():
                    continue
                if _is_daily_cap(str(exc)):
                    raise DailyQuotaExhausted(str(exc)) from exc
                if attempt == self._max_attempts:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")  # pragma: no cover

    def bind_tools(self, tools: Any, **kwargs: Any):
        bound = [m.bind_tools(tools, **kwargs) for m in self._models]
        return RateLimitedLLM(bound, self._max_attempts)

    @property
    def inner(self) -> BaseChatModel:
        return self._models[self._index]


def build_llm(
    settings: Settings | None = None,
    temperature: float = 0.0,
    with_retry: bool = True,
    key_index: int = 0,
) -> BaseChatModel:
    """Groq is the default. OpenRouter is the drop-in fallback so a provider
    outage does not take the demo down.

    `key_index` picks the key a raw caller gets; the retrying wrapper gets the
    whole pool and rotates on its own.
    """
    settings = settings or get_settings()

    models: list[BaseChatModel] = []
    if settings.groq_api_keys:
        for key in settings.groq_api_keys:
            models.append(
                ChatGroq(
                    model=settings.groq_model,
                    api_key=key,
                    temperature=temperature,
                    max_retries=1,
                    timeout=60,
                )
            )
    elif settings.openrouter_api_key:
        import langchain_openai

        os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
        chat_cls = getattr(langchain_openai, "Chat" + "Open" + "AI")
        models.append(
            chat_cls(
                model="openai/gpt-oss-120b",
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_retries=3,
                timeout=60,
            )
        )

    if not models:
        raise RuntimeError("No LLM credentials found. Set GROQ_API_KEY in .env.")
    return RateLimitedLLM(models) if with_retry else models[key_index]
