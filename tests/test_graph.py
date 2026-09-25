from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from conftest import FakeLLM
from langchain_core.messages import HumanMessage

from faq_service.application.workflows.graph import build_graph
from faq_service.domain.entities.document import RetrievedChunk
from faq_service.domain.errors import ServiceError


def initial():
    return {
        "messages": [HumanMessage(content="Как повторить выплату?")],
        "question": "Как повторить выплату?",
        "moderation": None,
        "searches": [],
        "evidence": [],
        "tool_calls": 0,
        "result": None,
    }


def repository():
    return SimpleNamespace(
        documents=AsyncMock(
            search=AsyncMock(
                return_value=[
                    RetrievedChunk(
                        id="chunk1",
                        document_id="doc1",
                        title="Выплата",
                        text="Нажмите Повторить.",
                        distance=0.1,
                    )
                ]
            )
        )
    )


async def test_full_graph_uses_tool_and_grounded_writer(settings, embeddings):
    db, llm = repository(), FakeLLM()
    graph = build_graph(settings, db, llm, embeddings)
    state = await graph.ainvoke(initial())
    assert state["result"].status == "answered"
    assert state["result"].citations[0].chunk_id == "chunk1"
    assert state["tool_calls"] == 1
    assert db.documents.search.await_count == 1
    assert "tools" in graph.get_graph().nodes


@pytest.mark.parametrize(
    "decision, status",
    [
        ("reject", "rejected"),
        ("clarify", "needs_clarification"),
    ],
)
async def test_moderation_bypasses_search(settings, embeddings, decision, status):
    db = repository()
    state = await build_graph(settings, db, FakeLLM(decision), embeddings).ainvoke(initial())
    assert state["result"].status == status
    db.documents.search.assert_not_called()


async def test_repeated_calls_are_bounded(settings, embeddings):
    db = repository()
    state = await build_graph(settings, db, FakeLLM(repeat=True), embeddings).ainvoke(initial())
    assert db.documents.search.await_count == settings.max_tool_calls
    assert len(state["evidence"]) == 1
    assert len(state["searches"]) == settings.max_tool_calls


async def test_no_search_no_fact_answer(settings, embeddings):
    db = repository()
    state = await build_graph(settings, db, FakeLLM(skip_search=True), embeddings).ainvoke(
        initial()
    )
    assert state["result"].status == "no_answer"
    db.documents.search.assert_not_called()


async def test_empty_search(settings, embeddings):
    db = SimpleNamespace(documents=AsyncMock(search=AsyncMock(return_value=[])))
    state = await build_graph(settings, db, FakeLLM(), embeddings).ainvoke(initial())
    assert state["result"].status == "no_answer"


async def test_fabricated_citation_is_not_returned(settings, embeddings):
    llm = FakeLLM(bad_citation=True)
    with pytest.raises(ServiceError, match="проверяемый ответ"):
        await build_graph(settings, repository(), llm, embeddings).ainvoke(initial())
    assert llm.writer_calls == 2


async def test_tool_failure_is_not_no_answer(settings, embeddings):
    db = SimpleNamespace(
        documents=AsyncMock(search=AsyncMock(side_effect=ServiceError("db", "Unavailable")))
    )
    with pytest.raises(ServiceError):
        await build_graph(settings, db, FakeLLM(), embeddings).ainvoke(initial())
