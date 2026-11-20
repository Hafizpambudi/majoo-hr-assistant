"""Embedder contract and key-rotation tests, no network.

The regression these guard: google-genai treats a flat `contents=["a", "b"]` as
one multi-part content and returns a single embedding, so a 90-chunk batch would
silently index 1 vector.
"""

import pytest

from app.config import Settings
from app.ingestion.embedder import (
    GeminiEmbedder,
    LocalHashEmbedder,
    QuotaExhausted,
    build_embedder,
)


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeModels:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def embed_content(self, model, contents, config):
        self.calls.append(contents)
        if self.error:
            raise self.error
        # Mirrors the real API: one embedding per content entry.
        return _FakeResponse([_FakeEmbedding([0.1, 0.2, 0.3]) for _ in contents])


class _FakeClient:
    def __init__(self, error=None):
        self.models = _FakeModels(error)


def _settings(**kwargs) -> Settings:
    base = {
        "_env_file": None,
        "gemini_api_key_1": "k1",
        "gemini_api_key_2": "k2",
        "gemini_api_key_3": "k3",
    }
    base.update(kwargs)
    return Settings(**base)


def _embedder(clients=None, settings=None) -> GeminiEmbedder:
    settings = settings or _settings()
    embedder = GeminiEmbedder(settings)
    embedder._clients = clients or [_FakeClient() for _ in settings.gemini_api_keys]
    embedder._batch = 2
    embedder._rpm_limit = 10_000
    return embedder


def _quota_error(message: str):
    from google.genai.errors import ClientError

    return ClientError(429, {"error": {"code": 429, "message": message}}, None)


def test_one_embedding_per_document():
    embedder = _embedder()
    assert len(embedder.embed_documents(["a", "b", "c", "d", "e"])) == 5


def test_each_text_sent_as_its_own_content():
    embedder = _embedder()
    embedder.embed_documents(["a", "b", "c"])
    assert embedder._clients[0].models.calls == [[["a"], ["b"]], [["c"]]]


def test_query_returns_a_single_vector():
    assert _embedder().embed_query("pertanyaan") == [0.1, 0.2, 0.3]


def test_key_pool_order_and_dedup():
    settings = Settings(
        _env_file=None,
        gemini_api_key_1="k1",
        gemini_api_key_2="",
        gemini_api_key_3="k3",
        gemini_api_key="k3",
    )
    assert settings.gemini_api_keys == ["k1", "k3"]


def test_missing_keys_raise():
    # _env_file=None keeps the real .env out of the test.
    with pytest.raises(RuntimeError, match="No Gemini key configured"):
        GeminiEmbedder(Settings(_env_file=None, gemini_api_key=""))


def test_rotates_on_daily_quota():
    exhausted = _FakeClient(_quota_error("quota exceeded per day, limit: 1000"))
    healthy = _FakeClient()
    embedder = _embedder(clients=[exhausted, healthy, healthy])

    assert embedder.embed_query("cuti") == [0.1, 0.2, 0.3]
    assert embedder._active == 1
    assert 0 in embedder._exhausted


def test_retries_same_key_on_per_minute_quota(monkeypatch):
    sleeps = []
    monkeypatch.setattr("app.ingestion.embedder.time.sleep", sleeps.append)

    class FlakyOnce:
        def __init__(self):
            self.calls = 0

        def embed_content(self, model, contents, config):
            self.calls += 1
            if self.calls == 1:
                raise _quota_error("requests per minute limit exceeded, limit: 100")
            return _FakeResponse([_FakeEmbedding([0.5]) for _ in contents])

    client = _FakeClient()
    client.models = FlakyOnce()
    embedder = _embedder(clients=[client, _FakeClient()])

    assert embedder.embed_query("cuti") == [0.5]
    assert embedder._active == 0
    assert embedder._exhausted == set()
    assert 20 in sleeps


def test_all_keys_exhausted_raises():
    bad = [_FakeClient(_quota_error("per day limit")) for _ in range(3)]
    embedder = _embedder(clients=bad)
    with pytest.raises(QuotaExhausted):
        embedder.embed_query("cuti")


def test_non_quota_error_is_not_swallowed():
    embedder = _embedder(clients=[_FakeClient(ValueError("boom"))])
    with pytest.raises(ValueError, match="boom"):
        embedder.embed_query("cuti")


def test_local_embedder_is_deterministic_and_normalised():
    embedder = LocalHashEmbedder(dim=32)
    first = embedder.embed_query("cuti tahunan")
    assert first == embedder.embed_query("cuti tahunan")
    assert abs(sum(v * v for v in first) ** 0.5 - 1.0) < 1e-9


def test_provider_selection():
    assert isinstance(
        build_embedder(Settings(embedding_provider="local")), LocalHashEmbedder
    )


def test_concurrent_embedding_does_not_retire_every_key():
    # Ragas embeds from worker threads. Without a lock, two threads can each see
    # a 429 and each rotate, retiring the whole pool from one rate-limit event.
    import threading

    dead = _FakeClient(_quota_error("quota exceeded per day, limit: 1000"))
    embedder = _embedder(clients=[dead, _FakeClient()])

    results, errors = [], []

    def worker():
        try:
            results.append(embedder.embed_query("cuti"))
        except Exception as exc:  # noqa: BLE001 - recorded for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # One key is genuinely dead, one is healthy: every call must have succeeded
    # and only the dead key may be retired.
    assert errors == []
    assert len(results) == 6
    assert embedder._exhausted == {0}
