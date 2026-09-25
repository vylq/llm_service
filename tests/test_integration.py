import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from conftest import FakeLLM

from faq_service.api.main import create_app
from faq_service.application.ingestion.pipeline import ingest
from faq_service.application.services.chat_service import ChatService
from faq_service.application.workflows.graph import build_graph
from faq_service.domain.errors import ServiceError
from faq_service.infrastructure.db.database import Database

pytestmark = pytest.mark.integration


async def test_ingestion_cosine_search_and_idempotency(settings, database, dataset, embeddings):
    result = await ingest(settings, database, embeddings)
    assert result["documents"] == 2
    assert result["chunks"] == 2
    calls = embeddings.calls
    assert (await ingest(settings, database, embeddings))["status"] == "unchanged"
    assert calls == embeddings.calls
    found = await database.documents.search([1, 0, 0], 2)
    assert [c.distance for c in found] == pytest.approx([0, 1])
    assert "выплат" in found[0].title
    settings.max_cosine_distance = 0.5
    assert len(await database.documents.search([1, 0, 0], 2)) == 1


async def test_failed_rebuild_preserves_index(settings, database, dataset, embeddings):
    await ingest(settings, database, embeddings)
    old = await database.corpus_info()
    dataset.write_text(
        dataset.read_text()
        + "\n"
        + json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Ещё?"},
                    {"role": "assistant", "content": "Ответ."},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="--rebuild"):
        await ingest(settings, database, embeddings)

    class BadEmbedding:
        async def aembed_documents(self, texts):
            return [[0, 0, 0] for _ in texts]

    with pytest.raises(ServiceError):
        await ingest(settings, database, BadEmbedding(), rebuild=True)
    assert await database.corpus_info() == old
    assert len(await database.documents.search([1, 0, 0], 10)) == 2
    await ingest(settings, database, embeddings, rebuild=True)
    assert len(await database.documents.search([1, 0, 0], 10)) == 3


async def test_profile_mismatch(settings, database, dataset, embeddings):
    await ingest(settings, database, embeddings)
    settings.embedding_model = "different-model"
    with pytest.raises(ServiceError, match="пересчитать"):
        await database.check_ready()


async def test_chat_lock_and_failed_history(settings, database, dataset, embeddings):
    await ingest(settings, database, embeddings)
    chat_id = await database.chats.create()
    async with database.chats.session(chat_id):
        with pytest.raises(ServiceError) as exc:
            async with database.chats.session(chat_id):
                pass
        assert exc.value.status_code == 409
    service = ChatService(
        settings, database, build_graph(settings, database, FakeLLM(bad_citation=True), embeddings)
    )
    with pytest.raises(ServiceError):
        await service.answer(chat_id, "Как повторить выплату?")
    history = await database.chats.history(chat_id)
    assert history[0]["status"] == "failed"
    assert history[0]["question"] == "Как повторить выплату?"


async def test_api_dialogue_and_restart(settings, database, dataset, embeddings):
    await ingest(settings, database, embeddings)
    graph = build_graph(settings, database, FakeLLM(), embeddings)
    app = create_app(settings, database, graph)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            assert (await c.get("/health/ready")).status_code == 200
            created = await c.post("/api/v1/chats")
            assert created.status_code == 201
            chat_id = created.json()["chat_id"]
            path = f"/api/v1/chats/{chat_id}/messages"
            response = await c.post(path, json={"message": "Как повторить выплату?"})
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["status"] == "answered"
            assert body["documents"][0]["body"]
            assert body["citations"][0]["document_id"] == body["documents"][0]["id"]
            assert (await c.post(path, json={"message": "А ещё?"})).status_code == 200
            history = (await c.get(path)).json()["turns"]
            assert len(history) == 2
            assert all(t["status"] == "completed" for t in history)
            assert (await c.post(path, json={"message": "   "})).status_code == 422
            assert (
                await c.post(f"/api/v1/chats/{uuid4()}/messages", json={"message": "test"})
            ).status_code == 404
    restarted_db = Database(settings, database.dsn)
    stored = await restarted_db.chats.history(chat_id)
    assert len(stored) == 2
    assert stored[0]["response"] == body


async def test_deadline_keeps_question(settings, database, dataset, embeddings):
    await ingest(settings, database, embeddings)
    settings.request_timeout_seconds = 0.03

    class SlowGraph:
        async def ainvoke(self, *args, **kwargs):
            await asyncio.sleep(1)

    app = create_app(settings, database, SlowGraph())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            chat_id = (await c.post("/api/v1/chats")).json()["chat_id"]
            path = f"/api/v1/chats/{chat_id}/messages"
            response = await c.post(path, json={"message": "test"})
            assert response.status_code == 504
            assert (await c.get(path)).json()["turns"][0]["status"] == "failed"
