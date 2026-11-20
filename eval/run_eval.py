"""Evaluation harness.

Two layers, on purpose:

1. Deterministic checks that gate a release: tool routing, refusal behaviour,
   citation presence, latency. Cheap, stable, no judge in the loop.
2. Ragas metrics (faithfulness, answer relevancy, context precision/recall) for
   answer quality, run with the same LLM as judge.

    python -m eval.run_eval                 # deterministic only
    python -m eval.run_eval --ragas         # + Ragas quality metrics
    python -m eval.run_eval --limit 5       # smoke run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app.agent.graph import HRAgent
from app.agent.nodes import REFUSAL
from app.config import get_settings

console = Console()
OUT_DIR = Path(__file__).resolve().parent


def load_golden(limit: int | None = None) -> list[dict]:
    path = get_settings().golden_path
    items = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return items[:limit] if limit else items


CITATION_RE = re.compile(r"\[\d+\]")

# Phrases that count as "I don't know". The case requires the agent to say it
# does not know rather than invent an answer; any of these satisfies that. A
# tool reporting no record for an unknown employee is a refusal, not an answer.
REFUSAL_MARKERS = (
    "tidak tahu",
    "tidak ada di dokumen",
    "tidak ditemukan",
    "tidak ada data",
    "tidak memiliki informasi",
)


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    if REFUSAL.lower()[:40] in lowered:
        return True
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def has_citation(answer: str) -> bool:
    return bool(CITATION_RE.search(answer))


async def run_agent(agent: HRAgent, items: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in items:
        question = item["question"]
        if item.get("employee_id"):
            question = f"{question} (employee_id: {item['employee_id']})"

        try:
            started = time.perf_counter()
            result = await agent.ask(question)
            elapsed = (time.perf_counter() - started) * 1000
        except Exception as exc:  # noqa: BLE001 - keep partial progress
            # A provider quota stop must not throw away the answers already
            # collected, which are the expensive part of the run.
            console.print(f"  {item['id']} STOP {type(exc).__name__}: {str(exc)[:90]}")
            save_partial(rows)
            console.print(
                f"[yellow]saved {len(rows)} answers to eval/results.json; "
                f"rerun with --from-results to score them[/]"
            )
            raise

        called = {c["name"] for c in result.tool_calls}
        expected = set(item.get("expected_tools", []))
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": result.answer,
                "reference": item.get("reference", ""),
                "contexts": [c["content"] for c in result.contexts],
                "retrieved": result.sources,
                "expected_tools": sorted(expected),
                "called_tools": sorted(called),
                "tools_ok": expected.issubset(called),
                "expect_refusal": item["expect_refusal"],
                "refused": is_refusal(result.answer),
                "refusal_ok": is_refusal(result.answer) == item["expect_refusal"],
                "cited": has_citation(result.answer),
                "tool_only": bool(called) and not result.sources,
                "latency_ms": round(elapsed, 1),
                "run_id": result.run_id,
            }
        )
        console.print(
            f"  {item['id']} {'OK ' if rows[-1]['refusal_ok'] else 'MISS'} "
            f"{result.answer[:70]}"
        )
        save_partial(rows)
    return rows


def save_partial(rows: list[dict]) -> None:
    """Persist progress after each question.

    Never shrink the file: a run that dies on the first question would
    otherwise erase a complete previous run.
    """
    path = OUT_DIR / "results.json"
    if path.exists() and len(rows) == 0:
        return
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8")).get("rows", [])
        except (json.JSONDecodeError, OSError):
            previous = []
        if len(previous) > len(rows):
            return
    path.write_text(
        json.dumps(
            {"metrics": {}, "ragas": {}, "rows": rows}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )


def deterministic_report(rows: list[dict]) -> dict:
    latencies = sorted(r["latency_ms"] for r in rows)
    answered = [r for r in rows if not r["refused"]]
    # Tool-only answers carry no document citation by design, so counting them
    # would report a citation failure that is not one.
    documented = [r for r in answered if not r["tool_only"]]
    return {
        "questions": len(rows),
        "tool_routing_accuracy": round(
            sum(r["tools_ok"] for r in rows) / len(rows), 3
        ),
        "refusal_accuracy": round(sum(r["refusal_ok"] for r in rows) / len(rows), 3),
        "false_answer_rate": round(
            sum(1 for r in rows if r["expect_refusal"] and not r["refused"]) / len(rows), 3
        ),
        "false_refusal_rate": round(
            sum(1 for r in rows if not r["expect_refusal"] and r["refused"]) / len(rows), 3
        ),
        "citation_rate": round(
            sum(r["cited"] for r in documented) / len(documented), 3
        )
        if documented
        else 0.0,
        "documented_answers": len(documented),
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[int(len(latencies) * 0.95) - 1],
    }


def run_ragas(rows: list[dict]) -> dict:
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings.base import BaseRagasEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except ImportError as exc:
        console.print(f"[yellow]Ragas unavailable: {exc}[/]")
        return {}

    from app.ingestion.embedder import build_embedder
    from app.llm import build_llm

    class EmbedderAdapter(BaseRagasEmbeddings):
        def __init__(self) -> None:
            super().__init__()
            self._inner = build_embedder()

        def embed_query(self, text: str) -> list[float]:
            return self._inner.embed_query(text)

        async def aembed_query(self, text: str) -> list[float]:
            return self.embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._inner.embed_documents(texts)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_documents(texts)

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"] or ["(no context)"],
            reference=r["reference"],
        )
        for r in rows
        if not r["refused"]
    ]
    if not samples:
        return {}

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=LangchainLLMWrapper(build_llm()),
        embeddings=EmbedderAdapter(),
    )
    return {k: round(float(v), 3) for k, v in result.to_pandas().mean(numeric_only=True).items()}


def write_report(rows: list[dict], metrics: dict, ragas: dict) -> Path:
    path = OUT_DIR / "report.md"

    lines = [
        "# Evaluation report",
        "",
        "## Deterministic metrics",
        "",
        "| metric | value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in metrics.items()],
        "",
    ]
    if ragas:
        lines += [
            "## Ragas metrics (answered questions only)",
            "",
            "| metric | value |",
            "|---|---|",
            *[f"| {k} | {v} |" for k, v in ragas.items()],
            "",
        ]
    lines += [
        "## Per question",
        "",
        "| id | tools | refusal | answer |",
        "|---|---|---|---|",
        *[
            f"| {r['id']} | {'OK' if r['tools_ok'] else 'MISS'} | "
            f"{'OK' if r['refusal_ok'] else 'MISS'} | "
            f"{r['answer'][:110].replace('|', '/').replace(chr(10), ' ')} |"
            for r in rows
        ],
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "results.json").write_text(
        json.dumps({"metrics": metrics, "ragas": ragas, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


async def main_async(args) -> None:
    if args.from_results:
        # Re-score saved answers instead of re-running the agent. Ragas is the
        # expensive part and the answers do not change between scoring runs.
        saved = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))
        rows = saved["rows"]
        console.print(f"[bold]rescoring {len(rows)} saved answers[/]")
    else:
        items = load_golden(args.limit)
        console.print(f"[bold]evaluating {len(items)} questions[/]")
        agent = HRAgent()
        await agent.setup()
        rows = await run_agent(agent, items)

    metrics = deterministic_report(rows)
    console.print(json.dumps(metrics, indent=2))

    ragas_metrics = run_ragas(rows) if (args.ragas or args.from_results) else {}
    if ragas_metrics:
        console.print(json.dumps(ragas_metrics, indent=2))

    path = write_report(rows, metrics, ragas_metrics)
    console.print(f"[green]report written: {path}[/]")

    thresholds = {"refusal_accuracy": 0.9, "tool_routing_accuracy": 0.85}
    failed = [
        name for name, floor in thresholds.items() if metrics.get(name, 0) < floor
    ]
    if failed:
        console.print(f"[red]below threshold: {failed}[/]")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ragas", action="store_true")
    parser.add_argument(
        "--from-results",
        action="store_true",
        help="rescore eval/results.json without re-running the agent",
    )
    asyncio.run(main_async(parser.parse_args()))
