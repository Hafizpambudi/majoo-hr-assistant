"""Resume logic tests against a stubbed store, no database needed."""

import scripts.ingest as ingest


class _StubStore:
    def __init__(self, existing=()):
        self._existing = set(existing)
        self.upserted = []
        self.reset_called = False

    def ensure_schema(self):
        pass

    def reset(self):
        self.reset_called = True
        self._existing.clear()

    def existing_ids(self):
        return set(self._existing)

    def upsert(self, chunks, vectors):
        self.upserted.extend(c.chunk_id for c in chunks)
        return len(chunks)

    def count(self):
        return len(self._existing) + len(self.upserted)


class _StubEmbedder:
    dim = 3
    key_count = 3

    def __init__(self):
        self.embedded = []

    def embed_documents(self, texts):
        self.embedded.extend(texts)
        return [[0.0, 0.0, 0.0] for _ in texts]


def _run(monkeypatch, argv, existing=(), chunks=None):
    store = _StubStore(existing)
    embedder = _StubEmbedder()
    chunks = chunks or [_chunk(i) for i in range(6)]

    monkeypatch.setattr(ingest, "build_chunks", lambda prefix=None: (chunks, 1))
    monkeypatch.setattr(ingest, "VectorStore", lambda settings: store)
    monkeypatch.setattr(ingest, "build_embedder", lambda settings: embedder)
    monkeypatch.setattr("sys.argv", ["ingest", *argv])

    ingest.main()
    return store, embedder


def _chunk(i):
    from app.ingestion.chunker import Chunk

    return Chunk(
        chunk_id=f"c{i}",
        doc_id="doc",
        text=f"text {i}",
        heading="Doc > H",
        ordinal=i,
        metadata={"title": "Doc"},
    )


def test_resume_skips_already_indexed_chunks(monkeypatch):
    store, embedder = _run(monkeypatch, [], existing={"c0", "c1", "c2"})
    assert store.upserted == ["c3", "c4", "c5"]
    assert len(embedder.embedded) == 3
    assert store.reset_called is False


def test_reset_forces_full_rebuild(monkeypatch):
    store, embedder = _run(monkeypatch, ["--reset"], existing={"c0", "c1"})
    assert store.reset_called is True
    assert len(embedder.embedded) == 6


def test_no_resume_reembeds_everything(monkeypatch):
    store, embedder = _run(monkeypatch, ["--no-resume"], existing={"c0"})
    assert len(embedder.embedded) == 6


def test_nothing_pending_exits_cleanly(monkeypatch):
    store, embedder = _run(
        monkeypatch, [], existing={f"c{i}" for i in range(6)}
    )
    assert embedder.embedded == []
    assert store.upserted == []


def test_quota_exhausted_exits_zero_and_keeps_progress(monkeypatch):
    import pytest

    from app.ingestion.embedder import QuotaExhausted

    class _QuotaEmbedder(_StubEmbedder):
        def embed_documents(self, texts):
            raise QuotaExhausted("all 3 keys hit their daily quota")

    store = _StubStore()
    monkeypatch.setattr(
        ingest, "build_chunks", lambda prefix=None: ([_chunk(i) for i in range(6)], 1)
    )
    monkeypatch.setattr(ingest, "VectorStore", lambda settings: store)
    monkeypatch.setattr(ingest, "build_embedder", lambda settings: _QuotaEmbedder())
    monkeypatch.setattr("sys.argv", ["ingest"])

    with pytest.raises(SystemExit) as exc:
        ingest.main()
    assert exc.value.code == 0


def test_progress_total_is_not_double_counted(monkeypatch, capsys):
    import pytest

    from app.ingestion.embedder import QuotaExhausted

    class _QuotaEmbedder(_StubEmbedder):
        def embed_documents(self, texts):
            raise QuotaExhausted("quota")

    store = _StubStore(existing={"c0", "c1"})
    monkeypatch.setattr(
        ingest, "build_chunks", lambda prefix=None: ([_chunk(i) for i in range(6)], 1)
    )
    monkeypatch.setattr(ingest, "VectorStore", lambda settings: store)
    monkeypatch.setattr(ingest, "build_embedder", lambda settings: _QuotaEmbedder())
    monkeypatch.setattr("sys.argv", ["ingest"])

    with pytest.raises(SystemExit):
        ingest.main()
    # 6 chunks total, 2 already present: the denominator must stay 6, not 10.
    assert "/6 chunks" in capsys.readouterr().out
