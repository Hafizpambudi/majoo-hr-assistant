"""Heading-aware markdown chunking.

Splitting on headings first keeps a chunk inside one policy topic, which matters
more for HR answers than hitting an exact token count. Long sections are then
packed to `chunk_size` characters with overlap so a sentence that straddles a
boundary is still retrievable from at least one chunk.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    heading: str
    ordinal: int
    metadata: dict = field(default_factory=dict)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading_path, body) pairs."""
    sections: list[tuple[str, str]] = []
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        body = text[match.end() : end].strip()
        if body:
            sections.append((heading, body))
    return sections


def _pack(body: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs into windows of roughly `size` characters."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    windows: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > size:
            if current:
                windows.append(current)
                current = ""
            step = max(size - overlap, 1)
            for start in range(0, len(para), step):
                piece = para[start : start + size].strip()
                if piece:
                    windows.append(piece)
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= size:
            current = candidate
        else:
            windows.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        windows.append(current)
    return windows


def chunk_document(
    text: str,
    doc_id: str,
    title: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in _split_sections(text):
        label = f"{title} > {heading}" if heading else title
        for piece in _pack(body, chunk_size, chunk_overlap):
            ordinal = len(chunks)
            digest = hashlib.blake2b(
                f"{doc_id}:{ordinal}:{piece}".encode("utf-8"), digest_size=6
            ).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{ordinal:03d}-{digest}",
                    doc_id=doc_id,
                    text=piece,
                    heading=label,
                    ordinal=ordinal,
                )
            )
    return chunks
