import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from mock_openrouter import response_for
from openai import AsyncOpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from faq_service.application.workflows.graph import build_graph
from faq_service.domain.entities.document import RetrievedChunk
from faq_service.infrastructure.llm.factory import chat_model, embedding_model


@pytest.fixture
def tracing(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # Keep real Langfuse callbacks/serialization; replace only network export.
    monkeypatch.setattr(
        "langfuse._client.resource_manager.LangfuseSpanProcessor",
        lambda **kwargs: SimpleSpanProcessor(exporter),
    )
    key = "pk-lf-test-" + uuid4().hex
    client = Langfuse(public_key=key, secret_key="test", tracer_provider=provider)
    yield CallbackHandler(public_key=key), exporter
    client.shutdown()
    provider.shutdown()


async def test_actual_clients_serialize_openrouter_requests(settings, tracing):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        assert request.headers["authorization"] == "Bearer test"
        return httpx.Response(200, json=response_for(request.url.path, body))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = AsyncOpenAI(api_key="test", base_url=settings.chat_base_url, http_client=http)
        llm = chat_model(settings)
        llm.root_async_client = client
        llm.async_client = client.chat.completions
        embeddings = embedding_model(settings)
        embeddings.async_client = client.embeddings
        assert await embeddings.aembed_documents(["Выплата", "Поддержка"]) == [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        db = SimpleNamespace(
            documents=AsyncMock(
                search=AsyncMock(
                    return_value=[
                        RetrievedChunk(
                            id="c1",
                            document_id="d1",
                            title="Выплата",
                            text="Нажмите Повторить.",
                            distance=0,
                        )
                    ]
                )
            )
        )
        state = await build_graph(settings, db, llm, embeddings).ainvoke(
            {
                "messages": [HumanMessage(content="Как повторить выплату?")],
                "question": "Как повторить выплату?",
                "moderation": None,
                "searches": [],
                "evidence": [],
                "tool_calls": 0,
                "result": None,
            },
            config={"callbacks": [tracing[0]], "metadata": {"langfuse_session_id": "test-chat"}},
        )
    assert state["result"].status == "answered"
    chat_requests = [body for path, body in requests if path.endswith("/chat/completions")]
    assert len(chat_requests) == 4  # moderation, RAG, RAG after tool, writer
    assert all(r["provider"]["require_parameters"] for r in chat_requests)
    assert chat_requests[0]["response_format"]["type"] == "json_schema"
    assert chat_requests[1]["tools"][0]["function"]["name"] == "vector_search"
    spans = tracing[1].get_finished_spans()
    names = {span.name for span in spans}
    assert {"moderation", "rag", "writer", "vector_search"} <= names
    generations = [
        s for s in spans if s.attributes.get("langfuse.observation.type") == "generation"
    ]
    assert len(generations) == 4
    assert len({s.context.trace_id for s in spans}) == 1
