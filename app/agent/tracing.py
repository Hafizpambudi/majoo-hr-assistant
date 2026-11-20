"""Per-run tracing: one JSONL line per node visit.

The harness is only as good as its observability. Every decision - route, tool
call, retry, refusal - lands here with latency and token counts, so a bad answer
can be replayed step by step instead of guessed at.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import get_settings


def new_run_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@contextmanager
def span(node: str, payload: dict[str, Any] | None = None) -> Iterator[dict]:
    """Time a node and record its outcome even when it raises."""
    record: dict[str, Any] = {
        "ts": _now(),
        "node": node,
        "status": "ok",
        **(payload or {}),
    }
    started = time.perf_counter()
    try:
        yield record
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)


def usage_from(message: Any) -> dict[str, int]:
    meta = getattr(message, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    return {
        "input_tokens": int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
        "output_tokens": int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        ),
    }


def write_trace(run_id: str, question: str, events: list[dict], summary: dict) -> Path:
    settings = get_settings()
    settings.traces_dir.mkdir(parents=True, exist_ok=True)
    path = settings.traces_dir / f"{run_id}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"ts": _now(), "node": "run", "question": question, "run_id": run_id},
                ensure_ascii=False,
            )
            + "\n"
        )
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        handle.write(
            json.dumps({"ts": _now(), "node": "summary", **summary}, ensure_ascii=False)
            + "\n"
        )
    return path
