"""Chunk, embed, and index the corpus into pgvector.

    python -m scripts.ingest                 # resume: skip chunks already indexed
    python -m scripts.ingest --reset         # drop the table and rebuild
    python -m scripts.ingest --dry-run       # chunk only, report stats
    python -m scripts.ingest --prefix majo-hr  # limit to one corpus subtree

Resume is the default because embedding is the expensive step and the Gemini
free tier caps daily usage. A run that stops on a quota error leaves a usable
partial index and prints the command to continue.
"""

from __future__ import annotations

import argparse
import sys
import time

from rich.console import Console

from app.config import get_settings
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.embedder import QuotaExhausted, build_embedder
from app.ingestion.loaders import load_documents
from app.retrieval.store import VectorStore

console = Console()


def build_chunks(prefix: str | None = None) -> tuple[list[Chunk], int]:
    settings = get_settings()
    docs = load_documents(settings.docs_dir)
    if prefix:
        docs = [d for d in docs if d.doc_id.startswith(prefix)]
        if not docs:
            raise RuntimeError(f"No documents match prefix {prefix!r}")

    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(
            chunk_document(
                text=doc.text,
                doc_id=doc.doc_id,
                title=doc.title,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )
    for chunk in chunks:
        chunk.metadata = {"title": chunk.heading.split(" > ")[0]}
    return chunks, len(docs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true", help="drop the table first")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="re-embed every chunk even if it is already indexed",
    )
    parser.add_argument("--prefix", default=None, help="only docs under this prefix")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    chunks, doc_count = build_chunks(args.prefix)
    console.print(
        f"[bold]{doc_count}[/] documents -> [bold]{len(chunks)}[/] chunks "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )

    if args.dry_run:
        lengths = sorted(len(c.text) for c in chunks)
        console.print(
            f"chunk chars: min={lengths[0]} p50={lengths[len(lengths)//2]} "
            f"p95={lengths[int(len(lengths) * 0.95)]} max={lengths[-1]}"
        )
        return

    store = VectorStore(settings)
    store.ensure_schema()
    if args.reset:
        store.reset()
        console.print("[yellow]table dropped, rebuilding from zero[/]")

    expected = len(chunks)
    if not args.no_resume:
        already = store.existing_ids()
        pending = [c for c in chunks if c.chunk_id not in already]
        skipped = len(chunks) - len(pending)
        if skipped:
            console.print(
                f"[cyan]resumed:[/] {skipped} chunks already indexed, "
                f"{len(pending)} to go"
            )
        chunks = pending

    if not chunks:
        console.print(f"[green]nothing to do[/] - table holds {store.count()} chunks")
        return

    embedder = build_embedder(settings)
    batch_size = args.batch_size or settings.embedding_batch_size
    keys = getattr(embedder, "key_count", 1)
    console.print(
        f"embedding with [bold]{type(embedder).__name__}[/] dim={embedder.dim} "
        f"keys={keys} batch={batch_size}"
    )

    started = time.perf_counter()
    total = 0
    try:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embedder.embed_documents([c.text for c in batch])
            total += store.upsert(batch, vectors)
            console.print(f"  indexed {total}/{len(chunks)}", end="\r")
    except QuotaExhausted as exc:
        elapsed = time.perf_counter() - started
        console.print(f"\n[yellow]stopped:[/] {exc}")
        console.print(
            f"indexed {total} this run in {elapsed:.0f}s; "
            f"table holds {store.count()}/{expected} chunks"
        )
        console.print(
            "[yellow]progress is saved.[/] Continue later with:\n"
            "  python -m scripts.ingest"
        )
        sys.exit(0)

    elapsed = time.perf_counter() - started
    console.print(f"\n[green]indexed {total} chunks in {elapsed:.1f}s[/]")
    console.print(f"rows in table: {store.count()}")


if __name__ == "__main__":
    main()
