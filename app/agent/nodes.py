from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    ANSWER_PROMPT,
    ROUTER_PROMPT,
    VERIFY_CRITIQUE,
    VERIFY_PROMPT,
)
from app.agent.state import AgentState
from app.agent.tracing import span, usage_from
from app.retrieval.retriever import Retriever

REFUSAL = (
    "Maaf, saya tidak tahu. Informasi itu tidak ada di dokumen HR yang saya akses."
)

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
# Model output sometimes carries narrow no-break spaces and non-breaking
# hyphens that break copy-paste and log parsing.
UNICODE_NOISE = str.maketrans(
    {
        "\u202f": " ",
        "\u00a0": " ",
        "\u2011": "-",
        "\u2013": "-",
        # gpt-oss sometimes emits CJK citation brackets.
        "\u3010": "[",
        "\u3011": "]",
        "\uff3b": "[",
        "\uff3d": "]",
    }
)


def clean_text(text: str) -> str:
    cleaned = text.translate(UNICODE_NOISE)
    # Some models annotate citations with a source line range: [2†L1-L2] -> [2].
    cleaned = re.sub(r"\[(\d+)\s*[†*][^\]]*\]", r"[\1]", cleaned)
    # Tool transport ids must never look like a citation.
    cleaned = re.sub(r"\[lc_[0-9a-fA-F-]+\]", "", cleaned)
    return cleaned.strip()


def parse_json(text: str) -> dict[str, Any]:
    """Tolerant JSON reader for model output: strips fences, grabs first object."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = JSON_BLOCK.search(cleaned)
        if not match:
            raise
        return json.loads(match.group(0))


def _extract_text(content: Any) -> str:
    """Pull the text out of an MCP content payload.

    MCP tools return a list of content blocks shaped like
    `[{"type": "text", "text": "...", "id": "lc_..."}]`. The `id` is a transport
    artefact, but once it reaches the writer the model copies it into the answer
    as if it were a citation (`13 hari [lc_cc43...]`). Only the text survives.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _render_tool_payload(raw: Any) -> str:
    """Turn a tool's JSON payload into readable lines.

    Feeding raw JSON to the writer made it echo field names back to the user
    ("13 hari (annual_remaining)"), so the payload is flattened into
    `key: value` pairs and nested lists are expanded.
    """
    text = _extract_text(raw)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text

    if not isinstance(data, dict):
        return str(data)

    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    rendered = ", ".join(f"{k}={v}" for k, v in item.items())
                    lines.append(f"  - {rendered}")
                else:
                    lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _tool_context(messages: list) -> str:
    blocks = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            blocks.append(f"{msg.name}:\n{_render_tool_payload(msg.content)}")
    return "\n".join(blocks) if blocks else "(none)"


def _context_block(context: list[dict[str, Any]]) -> str:
    if not context:
        return "(no document sources retrieved)"
    return "\n\n".join(
        f"[{i}] {c['doc_id']} > {c['heading']}\n{c['content']}"
        for i, c in enumerate(context, start=1)
    )


def _event(state: AgentState, record: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("trace", []), record]


def make_plan_node(llm) -> Callable[[AgentState], dict]:
    def plan_node(state: AgentState) -> dict:
        with span("plan") as rec:
            response = llm.invoke(ROUTER_PROMPT.format(question=state["question"]))
            try:
                plan = parse_json(response.content)
            except (json.JSONDecodeError, ValueError):
                plan = {
                    "use_policy_search": True,
                    "use_tools": True,
                    "reason": "router output unparsable, fell back to both",
                }
            rec.update(
                {
                    "use_policy_search": plan.get("use_policy_search"),
                    "use_tools": plan.get("use_tools"),
                    "reason": plan.get("reason"),
                    **usage_from(response),
                }
            )
        return {"plan": plan, "steps": state.get("steps", 0) + 1, "trace": _event(state, rec)}

    return plan_node


def make_retrieve_node(retriever: Retriever) -> Callable[[AgentState], dict]:
    def retrieve_node(state: AgentState) -> dict:
        with span("retrieve") as rec:
            result = retriever.retrieve(state["question"])
            rec.update(
                {
                    "top_similarity": round(result.top_similarity, 4),
                    "has_evidence": result.has_evidence,
                    "chunks": [c.chunk_id for c in result.chunks],
                }
            )
        return {
            "context": [asdict(c) for c in result.chunks],
            "evidence": result.has_evidence,
            "steps": state.get("steps", 0) + 1,
            "trace": _event(state, rec),
        }

    return retrieve_node


def make_agent_node(llm_with_tools) -> Callable[[AgentState], dict]:
    def agent_node(state: AgentState) -> dict:
        with span("agent") as rec:
            messages = [SystemMessage(AGENT_SYSTEM_PROMPT), *state.get("messages", [])]
            response = llm_with_tools.invoke(messages)
            calls = getattr(response, "tool_calls", []) or []
            rec.update(
                {
                    "tool_calls": [
                        {"name": c["name"], "args": c.get("args", {})} for c in calls
                    ],
                    **usage_from(response),
                }
            )
        return {
            "messages": [response],
            "steps": state.get("steps", 0) + 1,
            "trace": _event(state, rec),
        }

    return agent_node


def make_tools_node(tool_node) -> Callable[..., Any]:
    async def tools_node(state: AgentState, config: RunnableConfig) -> dict:
        with span("tools") as rec:
            # LangGraph 1.x requires the run config for ToolNode; MCP tools are
            # async, so this node must be async too.
            result = await tool_node.ainvoke(state, config)
            produced = result.get("messages", [])
            rec.update(
                {
                    "results": [
                        {
                            "tool": m.name,
                            "status": m.status if hasattr(m, "status") else "ok",
                            "preview": str(m.content)[:200],
                        }
                        for m in produced
                        if isinstance(m, ToolMessage)
                    ]
                }
            )
        return {
            "messages": produced,
            "steps": state.get("steps", 0) + 1,
            "trace": _event(state, rec),
        }

    return tools_node


def make_answer_node(llm) -> Callable[[AgentState], dict]:
    def answer_node(state: AgentState) -> dict:
        with span("answer") as rec:
            verdict = state.get("verdict") or {}
            critique = ""
            if verdict.get("unsupported_claims"):
                critique = VERIFY_CRITIQUE.format(
                    claims="; ".join(verdict["unsupported_claims"])
                )
            prompt = ANSWER_PROMPT.format(
                context=_context_block(state.get("context", [])),
                tool_context=_tool_context(state.get("messages", [])),
                question=state["question"],
                critique=critique,
            )
            response = llm.invoke(prompt)
            draft = clean_text(str(response.content))
            rec.update({"draft_chars": len(draft), **usage_from(response)})
        return {
            "draft": draft,
            "final": draft,
            "steps": state.get("steps", 0) + 1,
            "trace": _event(state, rec),
        }

    return answer_node


def make_verify_node(llm, max_attempts: int = 2) -> Callable[[AgentState], dict]:
    def verify_node(state: AgentState) -> dict:
        draft = state.get("draft", "")

        # A refusal makes no factual claims, so there is nothing to check. Sending
        # it to the judge produces a false "ungrounded" verdict, which then burns
        # the retry budget and turns a correct refusal into a loop.
        if draft.strip() == REFUSAL:
            with span("verify") as rec:
                rec.update({"grounded": True, "skipped": "draft is a refusal"})
            return {
                "verdict": {"grounded": True, "reason": "refusal, nothing to verify"},
                "steps": state.get("steps", 0) + 1,
                "trace": _event(state, rec),
            }

        with span("verify") as rec:
            prompt = VERIFY_PROMPT.format(
                context=_context_block(state.get("context", [])),
                tool_context=_tool_context(state.get("messages", [])),
                draft=draft,
            )
            response = llm.invoke(prompt)
            try:
                verdict = parse_json(response.content)
            except (json.JSONDecodeError, ValueError):
                # An unparsable judge is treated as a pass: refusing on a
                # checker bug would make the assistant useless.
                verdict = {"grounded": True, "reason": "judge output unparsable"}

            attempts = state.get("verify_attempts", 0) + 1
            exhausted = attempts >= max_attempts
            rec.update(
                {
                    "grounded": verdict.get("grounded"),
                    "attempt": attempts,
                    "exhausted": exhausted,
                    **usage_from(response),
                }
            )

        update: dict[str, Any] = {
            "verdict": verdict,
            "verify_attempts": attempts,
            "steps": state.get("steps", 0) + 1,
            "trace": _event(state, rec),
        }
        if verdict.get("grounded") is False and exhausted:
            update["final"] = REFUSAL
            update["refusal"] = "groundedness check failed twice"
        return update

    return verify_node


def refuse_node(state: AgentState) -> dict:
    with span("refuse") as rec:
        reason = state.get("refusal") or "no supporting evidence"
        rec.update({"reason": reason})
    return {
        "final": state.get("final") or REFUSAL,
        "refusal": reason,
        "trace": _event(state, rec),
    }
