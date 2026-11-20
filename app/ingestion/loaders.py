from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".md", ".markdown", ".txt"}
FIRST_PARTY_PREFIX = "majo-hr/"


@dataclass
class SourceDoc:
    doc_id: str
    title: str
    path: Path
    text: str


def _title_from(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
        if line.strip():
            return line.strip()[:120]
    return fallback


def load_documents(docs_dir: Path) -> list[SourceDoc]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    docs: list[SourceDoc] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        rel = path.relative_to(docs_dir).with_suffix("")
        docs.append(
            SourceDoc(
                doc_id=str(rel).replace("\\", "/"),
                title=_title_from(text, path.stem),
                path=path,
                text=text,
            )
        )
    if not docs:
        raise RuntimeError(f"No markdown/txt documents found in {docs_dir}")
    # First-party policy documents first. If an ingest is interrupted by a quota
    # cap, the chunks that land are the ones the golden set tests, not whatever
    # happens to sort first alphabetically.
    docs.sort(key=lambda d: (not d.doc_id.startswith(FIRST_PARTY_PREFIX), d.doc_id))
    return docs
