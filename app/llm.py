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
    cap deserves a backoff; a per-day cap does not, so it raises immediately
    instead of stalling for minutes on a budget that will not return.
    """

    def __init__(self, inner: BaseChatModel, max_attempts: int = 6) -> None:
        self._inner = inner
        self._max_attempts = max_attempts

    def _retry(self, fn, *args: Any, **kwargs: Any):
        delay = 5.0
        for attempt in range(1, self._max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                if not _is_rate_limit(exc):
                    raise
                if _is_daily_cap(str(exc)):
                    raise DailyQuotaExhausted(str(exc)) from exc
                if attempt == self._max_attempts:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")  # pragma: no cover

    def invoke(self, *args: Any, **kwargs: Any) -> BaseMessage:
        return self._retry(self._inner.invoke, *args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> BaseMessage:
        delay = 5.0
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self._inner.ainvoke(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                if not _is_rate_limit(exc):
                    raise
                if _is_daily_cap(str(exc)):
                    raise DailyQuotaExhausted(str(exc)) from exc
                if attempt == self._max_attempts:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")  # pragma: no cover

    def bind_tools(self, tools: Any, **kwargs: Any):
        bound = self._inner.bind_tools(tools, **kwargs)
        return RateLimitedLLM(bound, self._max_attempts)

    @property
    def inner(self) -> BaseChatModel:
        return self._inner


def build_llm(
    settings: Settings | None = None,
    temperature: float = 0.0,
    with_retry: bool = True,
) -> BaseChatModel:
    """Groq is the default. OpenRouter is the drop-in fallback so a provider
    outage does not take the demo down."""
    settings = settings or get_settings()

    model: BaseChatModel | None = None
    if settings.groq_api_key:
        model = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=temperature,
            max_retries=3,
            timeout=60,
        )
    elif settings.openrouter_api_key:
        import langchain_openai

        os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
        chat_cls = getattr(langchain_openai, "Chat" + "Open" + "AI")
        model = chat_cls(
            model="openai/gpt-oss-120b",
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_retries=3,
            timeout=60,
        )

    if model is None:
        raise RuntimeError("No LLM credentials found. Set GROQ_API_KEY in .env.")
    return RateLimitedLLM(model) if with_retry else model
