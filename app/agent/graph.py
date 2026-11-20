"""The agent harness: an explicit, bounded LangGraph state machine.

Why a graph and not a ReAct string loop: every hop is a named node with its own
trace record, the step budget is a hard edge condition instead of a hope, and the
retry path is testable. That is the difference between an agent and a prompt.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent import nodes as n
from app.agent.state import AgentState
from app.agent.tracing import new_run_id, write_trace
from app.config import ROOT, Settings, get_settings
from app.llm import build_llm
from app.retrieval.retriever import Retriever

MCP_SERVER_NAME = "majo_hr"


@dataclass
class AgentAnswer:
    question: str
    answer: str
    refused: bool
    sources: list[str] = field(default_factory=list)
    contexts: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    run_id: str = ""
    trace_path: Path | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)


def mcp_server_config(settings: Settings) -> dict[str, dict[str, Any]]:
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONUNBUFFERED": "1",
    }
    return {
        MCP_SERVER_NAME: {
            "command": sys.executable,
            "args": ["-m", "app.mcp_server.server"],
            "transport": "stdio",
            "cwd": str(ROOT),
            "env": env,
        }
    }


async def load_mcp_tools(settings: Settings) -> list:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(mcp_server_config(settings))
    return await client.get_tools()


def build_graph(llm, retriever: Retriever, tools: list, max_steps: int):
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    tool_node = ToolNode(tools)

    def route_after_plan(state: AgentState) -> str:
        plan = state.get("plan") or {}
        if plan.get("use_policy_search"):
            return "retrieve"
        if plan.get("use_tools"):
            return "agent"
        return "refuse"

    def route_after_retrieve(state: AgentState) -> str:
        plan = state.get("plan") or {}
        if state.get("evidence"):
            return "agent" if plan.get("use_tools") else "answer"
        return "agent" if plan.get("use_tools") else "refuse"

    def route_after_agent(state: AgentState) -> str:
        last = (state.get("messages") or [None])[-1]
        wants_tools = bool(getattr(last, "tool_calls", None))
        if wants_tools and state.get("steps", 0) < max_steps:
            return "tools"
        return "answer"

    def route_after_verify(state: AgentState) -> str:
        verdict = state.get("verdict") or {}
        if verdict.get("grounded") is not False:
            return END
        attempts = state.get("verify_attempts", 0)
        if attempts >= 2:
            return "refuse"
        return "answer"

    graph = StateGraph(AgentState)
    graph.add_node("plan", n.make_plan_node(llm))
    graph.add_node("retrieve", n.make_retrieve_node(retriever))
    graph.add_node("agent", n.make_agent_node(llm_with_tools))
    graph.add_node("tools", n.make_tools_node(tool_node))
    graph.add_node("answer", n.make_answer_node(llm))
    graph.add_node("verify", n.make_verify_node(llm))
    graph.add_node("refuse", n.refuse_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"retrieve": "retrieve", "agent": "agent", "refuse": "refuse"},
    )
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"agent": "agent", "answer": "answer", "refuse": "refuse"},
    )
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "answer": "answer"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("answer", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"answer": "answer", "refuse": "refuse", END: END},
    )
    graph.add_edge("refuse", END)

    return graph.compile()


class HRAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: Retriever | None = None,
        llm=None,
        tools: list | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or Retriever(self.settings)
        self.llm = llm or build_llm(self.settings)
        self._tools = tools
        self._graph = None

    async def setup(self) -> None:
        if self._tools is None:
            self._tools = await load_mcp_tools(self.settings)
        self._graph = build_graph(
            self.llm, self.retriever, self._tools, self.settings.max_steps
        )

    async def ask(self, question: str, persist_trace: bool = True) -> AgentAnswer:
        if self._graph is None:
            await self.setup()

        run_id = new_run_id()
        state: AgentState = {
            "question": question,
            "messages": [HumanMessage(content=question)],
            "steps": 0,
            "trace": [],
            "verify_attempts": 0,
        }
        final_state = await self._graph.ainvoke(
            state,
            config={
                "recursion_limit": self.settings.max_steps * 3 + 12,
                "run_name": f"hr-agent-{run_id}",
            },
        )

        trace = final_state.get("trace", [])
        answer_text = final_state.get("final") or n.REFUSAL
        tool_calls = [
            call
            for event in trace
            if event.get("node") == "agent"
            for call in event.get("tool_calls", [])
        ]
        sources = [c["doc_id"] for c in final_state.get("context", [])]

        path = None
        if persist_trace:
            path = write_trace(
                run_id,
                question,
                trace,
                {
                    "steps": final_state.get("steps", 0),
                    "refused": answer_text == n.REFUSAL,
                    "tool_calls": [c["name"] for c in tool_calls],
                },
            )

        return AgentAnswer(
            question=question,
            answer=answer_text,
            refused=answer_text == n.REFUSAL,
            sources=sorted(set(sources)),
            contexts=final_state.get("context", []),
            tool_calls=tool_calls,
            steps=final_state.get("steps", 0),
            run_id=run_id,
            trace_path=path,
            trace=trace,
            plan=final_state.get("plan", {}),
        )
