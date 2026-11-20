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


def run_ragas(
    rows: list[dict],
    limit: int | None = None,
    metric_filter: str | None = None,
) -> dict:
    try:
        from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
        from ragas.embeddings.base import BaseRagasEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            Faithfulness,
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
    if limit:
        samples = samples[:limit]
    if not samples:
        return {}

    dataset = EvaluationDataset(samples=samples)
    # Ragas drives the judge through the LangChain model interface, so it needs
    # the raw BaseChatModel. Our retry wrapper is not a chat model subclass and
    # would fail on `agenerate_prompt`. ChatGroq's own max_retries covers the
    # per-minute 429s that Ragas can hit while scoring 25 answers.
    #
    # Three Groq constraints shaped these settings:
    #   * answer_relevancy asks for `strictness` generations per sample (3 by
    #     default). Groq rejects n > 1 with a 400, so strictness drops to 1.
    #   * gpt-oss reasons before answering. A judge prompt over five retrieved
    #     chunks ran out of budget mid-thought (LLMDidNotFinishException), so
    #     reasoning effort is lowered.
    #   * the fallback key is capped at 8k tokens per minute, so the judge gets
    #     the last key in the pool and a token ceiling that fits one request.
    base = build_llm(with_retry=False, key_index=-1)
    overrides = {
        "max_tokens": 1536,
        "reasoning_effort": "low",
        "max_retries": 8,
        "timeout": 180,
    }
    judge_model = (
        base.model_copy(update=overrides) if hasattr(base, "model_copy") else base
    )
    judge = LangchainLLMWrapper(judge_model)
    metrics = [
        Faithfulness(),
        AnswerRelevancy(strictness=1, embeddings=EmbedderAdapter(), llm=judge),
        ContextPrecision(),
    ]
    if metric_filter is not None:
        keep = {name.strip() for name in metric_filter.split(",") if name.strip()}
        metrics = [m for m in metrics if m.name in keep]
    if not metrics:
        return {}
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge,
            embeddings=EmbedderAdapter(),
            # One worker: the free tier meters tokens per minute, and Ragas
            # prompts carry five retrieved chunks each. Concurrency just moves
            # the 429 from the request to the retry loop.
            run_config=RunConfig(timeout=300, max_workers=1),
        )
    except Exception as exc:  # noqa: BLE001 - a spent quota is not a crash
        console.print(f"[yellow]Ragas scoring aborted: {type(exc).__name__}[/]")
        return {}
    frame = result.to_pandas()
    metric_names = ["faithfulness", "answer_relevancy", "context_precision"]
    scores: dict = {}
    for name in metric_names:
        if name in frame.columns:
            # Ragas averages over the samples it managed to score. A mean
            # without its denominator hides a run that only scored half the set,
            # and a spent embedding quota drops answer_relevancy rows silently.
            scores[name] = round(float(frame[name].mean()), 3)
            scores[f"{name}_scored"] = int(frame[name].notna().sum())
    scores["total_samples"] = len(samples)
    return scores


def write_report(
    rows: list[dict],
    metrics: dict,
    ragas: dict,
    run: dict | None = None,
) -> Path:
    path = OUT_DIR / "report.md"

    from app.config import get_settings

    settings = get_settings()
    run = run or {}
    model = run.get("model", settings.groq_model)
    embeddings = run.get("embeddings", settings.embedding_model)
    lines = [
        "# Evaluation report",
        "",
        f"Model `{model}`, embeddings `{embeddings}` "
        f"({settings.embedding_dim}d), retrieval scope `{settings.source_scope or 'all'}`.",
        "",
        "## Deterministic metrics",
        "",
        "| metric | value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in metrics.items()],
        "",
    ]
    if ragas:
        total = ragas.get("total_samples", 0)
        names = [
            name
            for name in ("faithfulness", "answer_relevancy", "context_precision")
            if name in ragas
        ]
        lines += [
            "## Ragas metrics (answered questions only)",
            "",
            "| metric | value | scored |",
            "|---|---|---|",
            *[
                f"| {name} | {ragas[name]} | {ragas.get(f'{name}_scored', 0)}/{total} |"
                for name in names
            ],
            "",
        ]
        partial = [
            name
            for name in names
            if ragas.get(f"{name}_scored", 0) < total
        ]
        if partial:
            lines += [
                f"`{', '.join(partial)}` did not score every sample; the audit "
                "trail is in `eval/results.json`.",
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
        json.dumps(
            {"run": run, "metrics": metrics, "ragas": ragas, "rows": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


async def main_async(args) -> None:
    from app.config import get_settings

    settings = get_settings()
    run = {"model": settings.groq_model, "embeddings": settings.embedding_model}
    if args.from_results:
        # Re-score saved answers instead of re-running the agent. Ragas is the
        # expensive part and the answers do not change between scoring runs.
        saved = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))
        rows = saved["rows"]
        # Keep the provenance of the answers, which may predate this run.
        run = saved.get("run") or run
        console.print(f"[bold]rescoring {len(rows)} saved answers[/]")
    else:
        items = load_golden(args.limit)
        console.print(f"[bold]evaluating {len(items)} questions[/]")
        agent = HRAgent()
        await agent.setup()
        rows = await run_agent(agent, items)

    metrics = deterministic_report(rows)
    console.print(json.dumps(metrics, indent=2))

    ragas_metrics = (
        run_ragas(rows, limit=args.ragas_limit, metric_filter=args.ragas_metrics)
        if args.ragas
        else {}
    )
    if ragas_metrics:
        console.print(json.dumps(ragas_metrics, indent=2))

    path = write_report(rows, metrics, ragas_metrics, run)
    console.print(f"[green]report written: {path}[/]")

    # Ragas leaves httpx transports for the garbage collector. Collecting them
    # while the loop is alive avoids "__del__ ... Event loop is closed" noise
    # after a run that actually succeeded.
    import gc

    gc.collect()

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
        "--ragas-limit",
        type=int,
        default=None,
        help="score only the first N answered questions with Ragas",
    )
    parser.add_argument(
        "--ragas-metrics",
        default=None,
        help="comma separated subset of faithfulness,answer_relevancy,context_precision",
    )
    parser.add_argument(
        "--from-results",
        action="store_true",
        help="rescore eval/results.json without re-running the agent",
    )
    asyncio.run(main_async(parser.parse_args()))
