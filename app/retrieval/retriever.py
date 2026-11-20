from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.ingestion.embedder import Embedder, build_embedder
from app.retrieval.store import RetrievedChunk, VectorStore


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    has_evidence: bool
    top_similarity: float

    def as_context(self) -> str:
        blocks = []
        for i, c in enumerate(self.chunks, start=1):
            blocks.append(f"[{i}] {c.doc_id} > {c.heading}\n{c.content}")
        return "\n\n".join(blocks)


class Retriever:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or build_embedder(self.settings)
        self.store = store or VectorStore(self.settings)
        # The offline hash embedder lives on a much lower cosine scale than
        # Gemini, so the same threshold would refuse every offline query.
        self.threshold = (
            min(self.settings.similarity_threshold, 0.20)
            if self.settings.embedding_provider.lower() == "local"
            else self.settings.similarity_threshold
        )

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        query_vec = self.embedder.embed_query(query)
        chunks = self.store.search(
            query_vec, query, top_k=top_k, scope=self.settings.source_scope
        )

        threshold = self.threshold
        # A lexical hit means the query terms literally appear in the chunk, so
        # it clears the gate even when the dense score is weak.
        has_evidence = any(
            c.similarity >= threshold or c.lexical for c in chunks
        )
        top_similarity = max((c.similarity for c in chunks), default=0.0)
        return RetrievalResult(
            query=query,
            chunks=chunks,
            has_evidence=has_evidence,
            top_similarity=top_similarity,
        )
