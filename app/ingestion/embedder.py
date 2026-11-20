"""Embedding providers behind one small interface.

Gemini is the default because the corpus mixes Bahasa Indonesia and English and
Gemini's multilingual retrieval quality is the strongest of the hosted options
tried. A local ONNX provider is kept as a fallback so the pipeline stays
runnable offline and in CI without an API key.
"""

from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from collections import deque

from app.config import Settings, get_settings


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class QuotaExhausted(RuntimeError):
    """Every key in the pool has spent its daily quota."""


class GeminiEmbedder(Embedder):
    def __init__(self, settings: Settings) -> None:
        keys = settings.gemini_api_keys
        if not keys:
            raise RuntimeError(
                "No Gemini key configured. Set GEMINI_API_KEY_1..N or "
                "GEMINI_API_KEY in .env, or set EMBEDDING_PROVIDER=local."
            )
        from google import genai

        self._keys = keys
        self._clients = [genai.Client(api_key=k) for k in keys]
        self._active = 0
        self._exhausted: set[int] = set()
        self._model = settings.embedding_model
        self.dim = settings.embedding_dim
        self._batch = settings.embedding_batch_size
        self._rpm_limit = settings.embedding_rpm_limit
        self._sent: deque[tuple[float, int]] = deque()
        # The rate window and the active-key pointer are shared state. Ragas
        # embeds from several worker threads, and without this two threads can
        # each see a 429, each rotate, and retire every key in the pool.
        self._lock = threading.Lock()

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def _rotate(self, index: int, reason: str) -> None:
        """Retire the current key and move to the next usable one."""
        self._exhausted.add(index)
        remaining = [
            i for i in range(len(self._keys)) if i not in self._exhausted
        ]
        if not remaining:
            raise QuotaExhausted(
                f"all {len(self._keys)} Gemini key(s) hit their daily quota ({reason})"
            )
        self._active = remaining[0]
        self._sent.clear()

    def _throttle(self, count: int) -> None:
        """Stay under the per-minute quota.

        The free tier meters *content items*, not HTTP requests: one batch of 90
        texts costs 90 of the 100 items allowed per minute. Counting items here
        is what keeps a large ingest from dying on a 429.
        """
        while True:
            now = time.monotonic()
            while self._sent and now - self._sent[0][0] >= 60:
                self._sent.popleft()
            used = sum(n for _, n in self._sent)
            if used + count <= self._rpm_limit:
                self._sent.append((now, count))
                return
            wait = 60 - (now - self._sent[0][0]) + 0.5
            time.sleep(max(wait, 0.5))

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google import genai
        from google.genai import types

        with self._lock:
            attempts = 0
            while True:
                attempts += 1
                if attempts > len(self._keys) * 3 + 5:
                    raise QuotaExhausted("retry budget spent without a usable key")

                self._throttle(len(texts))
                try:
                    response = self._clients[self._active].models.embed_content(
                        model=self._model,
                        # Each text must be its own list: a flat list of strings
                        # is treated as one multi-part content and returns a
                        # single embedding.
                        contents=[[text] for text in texts],
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=self.dim,
                        ),
                    )
                except genai.errors.ClientError as exc:
                    message = str(exc)
                    if exc.code != 429:
                        raise
                    # The API returns 429 for both limits; only the message says
                    # which. A daily cap needs a different key, a per-minute cap
                    # just needs patience.
                    if "per day" in message or "PerDay" in message:
                        self._rotate(self._active, "daily")
                        continue
                    if "per minute" in message or "PerMinute" in message:
                        time.sleep(20)
                        continue
                    raise

                if len(response.embeddings) != len(texts):
                    raise RuntimeError(
                        f"expected {len(texts)} embeddings, got {len(response.embeddings)}"
                    )
                return [list(e.values) for e in response.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            out.extend(self._embed(batch, "RETRIEVAL_DOCUMENT"))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class LocalHashEmbedder(Embedder):
    """Deterministic bag-of-words fallback. No model download, no network.

    Quality is far below Gemini, so it exists only to keep tests and offline
    demos working. It is not a production choice.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    provider = settings.embedding_provider.lower()
    if provider == "gemini":
        return GeminiEmbedder(settings)
    if provider == "local":
        return LocalHashEmbedder(dim=settings.embedding_dim)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")
