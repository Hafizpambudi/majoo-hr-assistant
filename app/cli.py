"""Interactive CLI.

    python -m app.cli ask "berapa sisa cuti EMP-001?"
    python -m app.cli chat
    python -m app.cli trace <run_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config import get_settings
from app.agent.graph import HRAgent

# Windows consoles default to cp1252 and blow up on model output that contains
# non-breaking hyphens or other Unicode punctuation.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()


def render(answer) -> None:
    style = "yellow" if answer.refused else "green"
    console.print(Panel(answer.answer, title="Jawaban", border_style=style))

    table = Table(show_header=True, header_style="bold")
    table.add_column("step")
    table.add_column("node")
    table.add_column("detail")
    table.add_column("ms", justify="right")
    for event in answer.trace:
        detail = (
            event.get("reason")
            or event.get("tool_calls")
            or event.get("top_similarity")
            or event.get("grounded")
            or ""
        )
        table.add_row(
            str(event.get("node")),
            event.get("status", ""),
            str(detail)[:70],
            str(event.get("latency_ms", "")),
        )
    console.print(table)

    if answer.sources:
        console.print(f"[dim]sources: {', '.join(answer.sources)}[/]")
    if answer.tool_calls:
        console.print(f"[dim]tools: {[c['name'] for c in answer.tool_calls]}[/]")
    if answer.trace_path:
        console.print(f"[dim]trace: {answer.trace_path}[/]")


async def _ask(question: str) -> None:
    agent = HRAgent()
    await agent.setup()
    render(await agent.ask(question))


async def _chat() -> None:
    agent = HRAgent()
    await agent.setup()
    console.print("[bold]Majoo HR assistant[/] - kosongkan untuk keluar.")
    while True:
        try:
            question = console.input("[bold cyan]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        render(await agent.ask(question))


def _show_trace(run_id: str) -> None:
    settings = get_settings()
    path = settings.traces_dir / f"{run_id}.jsonl"
    if not path.exists():
        console.print(f"[red]trace not found: {path}[/]")
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        console.print(json.dumps(json.loads(line), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="hr-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question", nargs="+")

    sub.add_parser("chat", help="interactive loop")
    trace = sub.add_parser("trace", help="print a saved trace")
    trace.add_argument("run_id")

    args = parser.parse_args()
    if args.command == "ask":
        asyncio.run(_ask(" ".join(args.question)))
    elif args.command == "chat":
        asyncio.run(_chat())
    else:
        _show_trace(args.run_id)


if __name__ == "__main__":
    main()
