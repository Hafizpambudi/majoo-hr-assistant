"""Routing and refusal tests against a scripted LLM, no network needed."""

from langchain_core.messages import AIMessage

from app.agent.graph import build_graph
from app.agent.nodes import REFUSAL


class ScriptedLLM:
    """Returns queued responses in order; every response is a plain AI message."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, *args, **kwargs):
        self.calls += 1
        if not self._responses:
            return AIMessage(content='{"grounded": true, "unsupported_claims": []}')
        return self._responses.pop(0)

    def bind_tools(self, tools):
        return self


class StubRetriever:
    def __init__(self, has_evidence: bool):
        self.has_evidence = has_evidence

    def retrieve(self, query, top_k=None):
        from app.retrieval.retriever import RetrievalResult
        from app.retrieval.store import RetrievedChunk

        chunks = []
        if self.has_evidence:
            chunks = [
                RetrievedChunk(
                    chunk_id="c1",
                    doc_id="majo-hr/01-kebijakan-cuti",
                    title="Kebijakan Cuti",
                    heading="Kebijakan Cuti > Carry Over",
                    content="Maksimal 5 hari carry over.",
                    similarity=0.81,
                    score=0.03,
                )
            ]
        return RetrievalResult(
            query=query, chunks=chunks, has_evidence=self.has_evidence,
            top_similarity=0.81 if chunks else 0.2,
        )


def _run(llm, retriever, tools=None):
    graph = build_graph(llm, retriever, tools or [], max_steps=4)
    return graph.invoke(
        {
            "question": "berapa batas carry over cuti?",
            "messages": [],
            "steps": 0,
            "trace": [],
            "verify_attempts": 0,
        },
        config={"recursion_limit": 20},
    )


def test_out_of_scope_question_refuses_without_calling_llm_answer():
    llm = ScriptedLLM(
        [AIMessage(content='{"use_policy_search": false, "use_tools": false, "reason": "unrelated"}')]
    )
    state = _run(llm, StubRetriever(has_evidence=False))
    assert state["final"] == REFUSAL
    assert any(e["node"] == "refuse" for e in state["trace"])
    assert llm.calls == 1


def test_no_evidence_after_retrieval_refuses():
    llm = ScriptedLLM(
        [AIMessage(content='{"use_policy_search": true, "use_tools": false, "reason": "policy"}')]
    )
    state = _run(llm, StubRetriever(has_evidence=False))
    assert state["final"] == REFUSAL
    assert state["steps"] == 2


def test_grounded_answer_is_returned():
    llm = ScriptedLLM(
        [
            AIMessage(content='{"use_policy_search": true, "use_tools": false, "reason": "policy"}'),
            AIMessage(content="Maksimal 5 hari [1]."),
            AIMessage(content='{"grounded": true, "unsupported_claims": [], "reason": "ok"}'),
        ]
    )
    state = _run(llm, StubRetriever(has_evidence=True))
    assert state["final"] == "Maksimal 5 hari [1]."
    assert state["steps"] == 4


def test_failed_verification_retries_then_refuses():
    llm = ScriptedLLM(
        [
            AIMessage(content='{"use_policy_search": true, "use_tools": false, "reason": "policy"}'),
            AIMessage(content="Cuti bisa 30 hari."),
            AIMessage(content='{"grounded": false, "unsupported_claims": ["30 hari"], "reason": "no source"}'),
            AIMessage(content="Cuti bisa 30 hari."),
            AIMessage(content='{"grounded": false, "unsupported_claims": ["30 hari"], "reason": "no source"}'),
        ]
    )
    state = _run(llm, StubRetriever(has_evidence=True))
    assert state["final"] == REFUSAL
    assert state["verify_attempts"] == 2


def test_refusal_is_not_sent_to_the_judge():
    # The model refusing is a valid outcome; the verifier must not treat it as
    # an ungrounded draft and burn the retry budget.
    llm = ScriptedLLM(
        [
            AIMessage(content='{"use_policy_search": true, "use_tools": false, "reason": "policy"}'),
            AIMessage(content=REFUSAL),
        ]
    )
    state = _run(llm, StubRetriever(has_evidence=True))
    assert state["final"] == REFUSAL
    assert llm.calls == 2
    verify = [e for e in state["trace"] if e["node"] == "verify"]
    assert verify[0].get("skipped") == "draft is a refusal"
