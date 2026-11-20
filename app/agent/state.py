from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    question: str
    plan: dict[str, Any]
    context: list[dict[str, Any]]
    evidence: bool
    messages: Annotated[list, add_messages]
    draft: str
    verdict: dict[str, Any]
    verify_attempts: int
    steps: int
    trace: list[dict[str, Any]]
    refusal: str
    final: str
