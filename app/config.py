from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    groq_api_key_1: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    # Free-tier orgs differ: one key may allow 200k tokens/day while another
    # allows only 8k tokens/minute. The pool rotates on a daily cap.
    groq_tpm_limit: int = 8000
    openrouter_api_key: str = ""

    # Ordered pool. Ingest rotates to the next key when one hits its daily
    # quota, so several free-tier keys behave like one larger budget.
    gemini_api_key: str = ""
    gemini_api_key_1: str = ""
    gemini_api_key_2: str = ""
    gemini_api_key_3: str = ""
    gemini_api_key_4: str = ""
    gemini_api_key_5: str = ""

    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 1536

    database_url: str = "postgresql://majo:majo@localhost:5433/majo_hr"

    docs_dir: Path = ROOT / "data" / "docs"
    traces_dir: Path = ROOT / "traces"
    golden_path: Path = ROOT / "data" / "golden" / "qa.jsonl"

    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    # Restrict retrieval to one corpus subtree. "majo-hr" keeps the third-party
    # handbook, which describes other countries' HR policy, out of answers.
    # Set to "" to search the whole index.
    source_scope: str = "majo-hr"
    # Below this cosine similarity the retriever reports "no evidence" and the
    # agent refuses instead of guessing.
    similarity_threshold: float = 0.62
    max_steps: int = 4
    embedding_batch_size: int = Field(default=90, ge=1, le=100)
    # The Gemini free tier meters content items, not requests. 95 leaves a
    # margin for the query embedding that shares the same quota.
    embedding_rpm_limit: int = 95

    @property
    def groq_api_keys(self) -> list[str]:
        """Ordered key pool. The plain key is tried first, then the numbered ones."""
        candidates = [self.groq_api_key, self.groq_api_key_1]
        ordered: list[str] = []
        for key in candidates:
            key = key.strip()
            if key and key not in ordered:
                ordered.append(key)
        return ordered

    @property
    def gemini_api_keys(self) -> list[str]:
        """Every configured key, numbered first, legacy single key last."""
        numbered = [
            self.gemini_api_key_1,
            self.gemini_api_key_2,
            self.gemini_api_key_3,
            self.gemini_api_key_4,
            self.gemini_api_key_5,
        ]
        ordered = [k.strip() for k in numbered if k.strip()]
        legacy = self.gemini_api_key.strip()
        if legacy and legacy not in ordered:
            ordered.append(legacy)
        return ordered


@lru_cache
def get_settings() -> Settings:
    return Settings()
