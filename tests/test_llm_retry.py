"""Rate-limit wrapper tests. The Groq free tier 429s a long eval run, so a
throttled request must wait and retry, not abort the run."""

import pytest

from app.llm import DailyQuotaExhausted, RateLimitedLLM


class _Boom(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class _Flaky:
    def __init__(self, failures: int, exc: Exception):
        self.failures = failures
        self.exc = exc
        self.calls = 0

    def invoke(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return "ok"

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("app.llm.time.sleep", lambda _s: None)


def test_retries_a_429_then_succeeds():
    inner = _Flaky(2, _Boom("rate limit reached", status_code=429))
    assert RateLimitedLLM(inner).invoke("prompt") == "ok"
    assert inner.calls == 3


def test_retries_on_rate_limit_text_without_a_status_code():
    inner = _Flaky(1, _Boom("Rate limit reached for model"))
    assert RateLimitedLLM(inner).invoke("prompt") == "ok"


def test_non_rate_limit_error_propagates_immediately():
    inner = _Flaky(1, _Boom("bad request", status_code=400))
    with pytest.raises(_Boom):
        RateLimitedLLM(inner).invoke("prompt")
    assert inner.calls == 1


def test_gives_up_after_max_attempts():
    inner = _Flaky(99, _Boom("rate limit", status_code=429))
    with pytest.raises(_Boom):
        RateLimitedLLM(inner, max_attempts=3).invoke("prompt")
    assert inner.calls == 3


def test_daily_cap_fails_fast_without_retrying():
    # Waiting does not refill a daily budget, so retrying only stalls the run.
    inner = _Flaky(99, _Boom("rate limit on tokens per day (TPD): Limit 200000"))
    with pytest.raises(DailyQuotaExhausted):
        RateLimitedLLM(inner).invoke("prompt")
    assert inner.calls == 1


def test_daily_cap_is_detected_from_the_provider_message():
    exc = _Boom("Rate limit reached for model on tokens per day (TPD): Limit 200000")
    with pytest.raises(DailyQuotaExhausted):
        RateLimitedLLM(_Flaky(1, exc)).invoke("prompt")


def test_bind_tools_stays_retry_wrapped():
    wrapped = RateLimitedLLM(_Flaky(0, _Boom("x"))).bind_tools([])
    assert isinstance(wrapped, RateLimitedLLM)


def test_daily_cap_rotates_to_the_next_key():
    # A spent daily budget is recoverable if another key has its own budget.
    dead = _Flaky(99, _Boom("rate limit on tokens per day (TPD)", status_code=429))
    healthy = _Flaky(0, _Boom("unused"))
    llm = RateLimitedLLM([dead, healthy])
    assert llm.invoke("prompt") == "ok"
    assert dead.calls == 1
    assert healthy.calls == 1


def test_daily_cap_raises_when_every_key_is_spent():
    dead = _Flaky(99, _Boom("rate limit on tokens per day (TPD)", status_code=429))
    with pytest.raises(DailyQuotaExhausted):
        RateLimitedLLM([dead, dead]).invoke("prompt")
