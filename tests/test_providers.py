import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from mock_openrouter import response_for
from ollama import AsyncClient
from pydantic import SecretStr, ValidationError

from faq_service.application.workflows.graph import build_graph
from faq_service.domain.entities.document import RetrievedChunk
from faq_service.infrastructure.llm.factory import chat_model, embedding_model
from faq_service.settings.app_settings import Settings


def test_provider_selection_and_independent_embeddings(settings):
    assert isinstance(chat_model(settings), ChatOpenAI)
    assert isinstance(embedding_model(settings), OpenAIEmbeddings)
    previous_profile = settings.embedding_profile
    settings.llm_provider = "ollama"
    assert isinstance(chat_model(settings), ChatOllama)
    assert isinstance(embedding_model(settings), OpenAIEmbeddings)
    assert settings.embedding_profile == previous_profile
    settings.embedding_provider = "ollama"
    settings.openrouter_api_key = SecretStr("")
    assert isinstance(chat_model(settings), ChatOllama)
    assert isinstance(embedding_model(settings), OllamaEmbeddings)
    assert settings.embedding_profile != previous_profile
    assert settings.chat_base_url == "http://localhost:11434"
    assert settings.embedding_profile["model"] == "nomic-embed-text"


def test_unknown_provider_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="unknown")


async def test_native_ollama_transport_and_graph(settings):
    settings.llm_provider = "ollama"
    settings.embedding_provider = "ollama"
    settings.openrouter_api_key = SecretStr("")
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        assert "authorization" not in request.headers
        if request.url.path == "/api/embed":
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "embeddings": [[1.0, 0.0, 0.0] for _ in body["input"]],
                },
            )
        assert request.url.path == "/api/chat"
        converted = {"model": body["model"], "messages": body["messages"]}
        if body.get("format"):
            converted["response_format"] = {"json_schema": {"name": body["format"]["title"]}}
        if body.get("tools"):
            converted["tools"] = body["tools"]
        message = response_for("/chat/completions", converted)["choices"][0]["message"]
        message["content"] = message.get("content") or ""
        for call in message.get("tool_calls", []):
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        payload = {
            "model": body["model"],
            "message": message,
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 10,
        }
        return httpx.Response(200, content=json.dumps(payload) + "\n")

    client = AsyncClient(host=settings.ollama_base_url, transport=httpx.MockTransport(handle))
    llm, embeddings = chat_model(settings), embedding_model(settings)
    llm._async_client = client
    embeddings._async_client = client
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
    try:
        result = await build_graph(settings, db, llm, embeddings).ainvoke(
            {
                "messages": [HumanMessage(content="Как повторить выплату?")],
                "question": "Как повторить выплату?",
                "moderation": None,
                "searches": [],
                "evidence": [],
                "tool_calls": 0,
                "result": None,
            }
        )
        assert result["result"].status == "answered"
        assert len([path for path, _ in requests if path == "/api/chat"]) == 4
        assert any(path == "/api/embed" for path, _ in requests)
    finally:
        await client._client.aclose()
