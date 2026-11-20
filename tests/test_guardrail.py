from app.config import Settings
from app.retrieval.retriever import Retriever
from app.retrieval.store import RetrievedChunk


class StubEmbedder:
    dim = 3

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


class StubStore:
    def __init__(self, chunks, scope=None):
        self._chunks = chunks
        self.scope = scope

    def search(self, query_vec, query_text, top_k=None, scope=None):
        self.scope = scope if scope is not None else self.scope
        return self._chunks


def _chunk(similarity: float, lexical: bool = False) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        doc_id="doc",
        title="Doc",
        heading="Doc > Cuti",
        content="isi",
        similarity=similarity,
        score=0.02,
        lexical=lexical,
    )


def _retriever(chunks) -> Retriever:
    settings = Settings(similarity_threshold=0.62)
    return Retriever(settings=settings, embedder=StubEmbedder(), store=StubStore(chunks))

def test_weak_dense_match_blocks_answer():
    result = _retriever([_chunk(0.31)]).retrieve("pertanyaan asing")
    assert result.has_evidence is False


def test_strong_dense_match_passes_gate():
    result = _retriever([_chunk(0.78)]).retrieve("berapa cuti tahunan")
    assert result.has_evidence is True


def test_lexical_hit_passes_gate_even_with_low_similarity():
    result = _retriever([_chunk(0.2, lexical=True)]).retrieve("BPJS")
    assert result.has_evidence is True


def test_empty_index_blocks_answer():
    result = _retriever([]).retrieve("apa pun")
    assert result.has_evidence is False
    assert result.top_similarity == 0.0


def test_retrieval_is_scoped_to_first_party_documents():
    # The third-party handbook describes other countries' HR policy, so the
    # default scope must exclude it.
    store = StubStore([_chunk(0.8)])
    retriever = Retriever(
        settings=Settings(similarity_threshold=0.62),
        embedder=StubEmbedder(),
        store=store,
    )
    retriever.retrieve("berapa lama cuti melahirkan")
    assert store.scope == "majo-hr"


def test_scope_can_be_disabled():
    store = StubStore([_chunk(0.8)])
    retriever = Retriever(
        settings=Settings(similarity_threshold=0.62, source_scope=""),
        embedder=StubEmbedder(),
        store=store,
    )
    retriever.retrieve("cuti")
    assert store.scope == ""
