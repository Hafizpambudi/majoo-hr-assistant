"""pgvector storage with hybrid (dense + lexical) retrieval fused by RRF.

Hybrid was worth the extra ~30 lines: HR questions mix paraphrases ("boleh
enggak cuti dipindah?") with exact policy terms ("BPJS", "PDP", "F1 visa") that
dense embeddings blur. Reciprocal Rank Fusion merges both rankings without
tuning a weight.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.config import Settings, get_settings
from app.ingestion.chunker import Chunk

RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    heading: str
    content: str
    similarity: float
    score: float
    lexical: bool = False

    @property
    def citation(self) -> str:
        return f"{self.doc_id} :: {self.heading}"


class VectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dim = self.settings.embedding_dim

    def connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.settings.database_url, row_factory=dict_row)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id  TEXT PRIMARY KEY,
                    doc_id    TEXT NOT NULL,
                    title     TEXT NOT NULL,
                    heading   TEXT NOT NULL,
                    ordinal   INTEGER NOT NULL,
                    content   TEXT NOT NULL,
                    embedding vector({self.dim}) NOT NULL,
                    metadata  JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_lexical_idx "
                "ON chunks USING gin (to_tsvector('simple', content))"
            )

    def reset(self) -> None:
        with self.connect() as conn:
            conn.execute("DROP TABLE IF EXISTS chunks")
        self.ensure_schema()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        rows = [
            (
                c.chunk_id,
                c.doc_id,
                c.metadata.get("title", ""),
                c.heading,
                c.ordinal,
                c.text,
                json.dumps(c.metadata),
                Vector(emb),
            )
            for c, emb in zip(chunks, embeddings)
        ]
        with self.connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (chunk_id, doc_id, title, heading, ordinal, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                rows,
            )
        return len(rows)

    def count(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def existing_ids(self) -> set[str]:
        """Chunk ids already indexed, so a re-run can skip them."""
        with self.connect() as conn:
            rows = conn.execute("SELECT chunk_id FROM chunks").fetchall()
        return {row["chunk_id"] for row in rows}

    def _dense(self, conn, query_vec: list[float], limit: int, scope: str | None) -> list[dict]:
        vec = Vector(query_vec)
        where, params = self._scope_clause(scope)
        return conn.execute(
            f"""
            SELECT chunk_id, doc_id, title, heading, content,
                   1 - (embedding <=> %s) AS similarity
            FROM chunks
            {where}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (vec, *params, vec, limit),
        ).fetchall()

    def _lexical(self, conn, query: str, limit: int, scope: str | None) -> list[dict]:
        where, params = self._scope_clause(scope)
        joiner = "AND" if where else "WHERE"
        return conn.execute(
            f"""
            SELECT chunk_id, doc_id, title, heading, content,
                   0.0 AS similarity
            FROM chunks
            {where}
            {joiner} to_tsvector('simple', content) @@ plainto_tsquery('simple', %s)
            ORDER BY ts_rank(to_tsvector('simple', content),
                             plainto_tsquery('simple', %s)) DESC
            LIMIT %s
            """,
            (*params, query, query, limit),
        ).fetchall()

    def _scope_clause(self, scope: str | None) -> tuple[str, tuple]:
        """Restrict retrieval to one corpus subtree.

        Default is first-party: the third-party handbook is a different company
        in different countries, so it contains HR policy that directly
        contradicts the Majo documents (for example a 4-week Danish maternity
        leave next to Majo's 3 months). Retrieving both and letting the model
        choose produces confidently wrong answers.
        """
        if not scope:
            return "", ()
        return "WHERE doc_id LIKE %s", (f"{scope}%",)

    def search(
        self,
        query_vec: list[float],
        query_text: str,
        top_k: int | None = None,
        scope: str | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.top_k
        pool = max(top_k * 3, 12)
        scope = scope if scope is not None else self.settings.source_scope

        with self.connect() as conn:
            dense = self._dense(conn, query_vec, pool, scope)
            try:
                lexical = self._lexical(conn, query_text, pool, scope)
            except psycopg.errors.SyntaxError:
                lexical = []

        fused: dict[str, dict] = {}
        for name, ranking in (("dense", dense), ("lexical", lexical)):
            for rank, row in enumerate(ranking):
                entry = fused.setdefault(
                    row["chunk_id"], {**row, "score": 0.0, "lexical": False}
                )
                entry["score"] += 1.0 / (RRF_K + rank + 1)
                entry["similarity"] = max(entry["similarity"], row["similarity"])
                entry["lexical"] = entry["lexical"] or name == "lexical"

        ordered = sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                title=r["title"],
                heading=r["heading"],
                content=r["content"],
                similarity=float(r["similarity"]),
                score=float(r["score"]),
                lexical=bool(r["lexical"]),
            )
            for r in ordered
        ]
