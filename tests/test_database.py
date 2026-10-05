"""Regression tests for transaction and lock semantics on PostgreSQL/pgvector."""

import asyncio
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import DDL, column, func, insert, select, table
from sqlalchemy.exc import IntegrityError, OperationalError

from faq_service.api.main import create_app
from faq_service.application.ingestion.pipeline import ingest, load_documents
from faq_service.application.services.chat_service import ChatService
from faq_service.domain.errors import ServiceError
from faq_service.infrastructure.db.locks import CORPUS_LOCK, INGEST_LOCK, advisory_lock
from faq_service.infrastructure.db.models import Base, CorpusRow, TurnRow

pytestmark = pytest.mark.integration


async def idle_transaction_count(database):
    activity = table("pg_stat_activity", column("datname"), column("pid"), column("state"))
    async with database.engine.begin() as conn:
        return await conn.scalar(
            select(func.count())
            .select_from(activity)
            .where(
                activity.c.datname == func.current_database(),
                activity.c.pid != func.pg_backend_pid(),
                activity.c.state == "idle in transaction",
            )
        )


async def test_existing_schema_and_data_survive_initialize(settings, database, dataset, embeddings):
    # The immutable old DDL exercises compatibility with databases predating SQLAlchemy.
    legacy_schema = Path(__file__).with_name("fixtures").joinpath("legacy_schema.sql")
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.execute(DDL(legacy_schema.read_text()))

    await ingest(settings, database, embeddings)
    chat_id = await database.chats.create()
    async with database.chats.session(chat_id) as conn:
        run_id = await database.chats.begin_turn(conn, chat_id, "Старый вопрос")
        await database.chats.fail_turn(conn, run_id, "old_error")
    before = await database.corpus_info()
    await database.initialize()
    await database.initialize()
    assert await database.corpus_info() == before
    assert len(await database.documents.search([1, 0, 0], 10)) == 2
    history = await database.chats.history(chat_id)
    assert history[0]["error_code"] == "old_error"
    assert isinstance(history[0]["id"], UUID)
    assert isinstance(history[0]["created_at"], datetime)
    assert history[0]["created_at"].tzinfo is not None


async def test_corpus_write_failure_rolls_back_documents_chunks_and_manifest(
    settings, database, dataset, embeddings
):
    await ingest(settings, database, embeddings)
    previous_info = await database.corpus_info()
    previous_chunks = await database.documents.search([1, 0, 0], 10)
    documents, _, _ = load_documents(dataset)
    # The second batch violates the FK after deletion and a successful first batch.
    chunks = [
        {"id": "new", "document_id": documents[0].id, "text": "new", "embedding": [1, 0, 0]},
        {"id": "broken", "document_id": "missing", "text": "bad", "embedding": [1, 0, 0]},
    ]
    async with database.corpus.ingestion() as conn:
        with pytest.raises(IntegrityError):
            await database.corpus.replace(
                conn, documents, chunks, "changed", settings.embedding_profile, batch_size=1
            )
        assert not conn.in_transaction()
    assert await database.corpus_info() == previous_info
    assert await database.documents.search([1, 0, 0], 10) == previous_chunks


async def test_document_order_duplicates_and_missing_ids(settings, database, dataset, embeddings):
    await ingest(settings, database, embeddings)
    chunks = await database.documents.search([1, 0, 0], 10)
    ids = [chunk.document_id for chunk in chunks]
    found = await database.documents.get_many([ids[1], "missing", ids[0], ids[1]])
    assert [document.id for document in found] == [ids[1], ids[0]]
    assert found[0].sources[0].file == dataset.name
    assert await database.documents.get_many([]) == []


async def test_chat_session_commits_pending_and_recovers_interrupted_turn(database):
    chat_id = await database.chats.create()
    async with database.chats.session(chat_id) as conn:
        assert not conn.in_transaction()
        interrupted = await database.chats.begin_turn(conn, chat_id, "Первый")
        assert not conn.in_transaction()
        pending = await database.chats.history(chat_id)
        assert pending[0]["status"] == "pending"

    async with database.chats.session(chat_id) as conn:
        run_id = await database.chats.begin_turn(conn, chat_id, "Второй")
        await database.chats.fail_turn(conn, run_id, "test_error")
        assert not conn.in_transaction()
    history = await database.chats.history(chat_id)
    by_id = {turn["id"]: turn for turn in history}
    assert by_id[interrupted]["status"] == "failed"
    assert by_id[interrupted]["error_code"] == "interrupted"
    assert by_id[run_id]["error_code"] == "test_error"
    assert await database.chats.history(chat_id, limit=1, offset=1) == history[1:2]


async def test_cancelled_workflow_persists_failure_and_releases_locks(
    settings, database, dataset, embeddings
):
    await ingest(settings, database, embeddings)
    started = asyncio.Event()

    class WaitingGraph:
        async def ainvoke(self, *args, **kwargs):
            started.set()
            await asyncio.Event().wait()

    chat_id = await database.chats.create()
    service = ChatService(settings, database, WaitingGraph())
    task = asyncio.create_task(service.answer(chat_id, "Проверка отмены"))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        history = await database.chats.history(chat_id)
        assert history[0]["status"] == "pending"
        assert await idle_transaction_count(database) == 0
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await service.close()
    assert (await database.chats.history(chat_id))[0]["status"] == "failed"
    async with database.chats.session(chat_id):
        pass
    async with database.engine.connect() as conn:
        async with advisory_lock(conn, CORPUS_LOCK, AssertionError("Corpus lock leaked")):
            pass


async def test_partial_chat_lock_acquisition_is_cleaned_up(database):
    chat_id = await database.chats.create()
    async with database.engine.connect() as conn:
        async with advisory_lock(conn, CORPUS_LOCK, AssertionError("Could not lock corpus")):
            with pytest.raises(ServiceError) as exc:
                async with database.chats.session(chat_id):
                    pass
            assert exc.value.code == "index_updating"
    async with database.chats.session(chat_id):
        pass


async def test_embedding_calls_have_no_transaction_and_ingestion_is_exclusive(
    settings, database, dataset, embeddings
):
    class InspectingEmbeddings:
        async def aembed_documents(self, texts):
            assert await idle_transaction_count(database) == 0
            with pytest.raises(ValueError, match="Another ingestion is running"):
                await ingest(settings, database, embeddings)
            return await embeddings.aembed_documents(texts)

    result = await ingest(settings, database, InspectingEmbeddings())
    assert result["status"] == "imported"
    assert embeddings.calls == 1
    # An early return from unchanged ingestion must also release its lock.
    assert (await ingest(settings, database, embeddings))["status"] == "unchanged"
    async with database.corpus.ingestion():
        pass


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_session_lock_released_on_same_physical_connection(database, outcome):
    async with database.engine.connect() as first, database.engine.connect() as second:
        expected = {
            "success": nullcontext(),
            "error": pytest.raises(ValueError, match="test"),
            "cancel": pytest.raises(asyncio.CancelledError),
        }[outcome]
        with expected:
            async with advisory_lock(first, INGEST_LOCK, AssertionError("First lock failed")):
                assert not first.in_transaction()
                with pytest.raises(ValueError, match="busy"):
                    async with advisory_lock(second, INGEST_LOCK, ValueError("busy")):
                        pass
                if outcome == "error":
                    raise ValueError("test")
                if outcome == "cancel":
                    raise asyncio.CancelledError
        # first remains physically connected; closing it cannot hide a missing unlock.
        async with advisory_lock(second, INGEST_LOCK, AssertionError("Lock leaked")):
            assert not second.in_transaction()


async def test_rebuild_waits_for_active_chat(database, settings, dataset, embeddings):
    await ingest(settings, database, embeddings)
    documents, _, _ = load_documents(dataset)
    chunks = [
        {"id": "replacement", "document_id": documents[0].id, "text": "new", "embedding": [1, 0, 0]}
    ]
    chat_id = await database.chats.create()
    async with database.corpus.ingestion() as conn:
        async with database.chats.session(chat_id):
            replacement = asyncio.create_task(
                database.corpus.replace(
                    conn, documents, chunks, "new", settings.embedding_profile, batch_size=1
                )
            )
            try:
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(replacement), timeout=0.05)
                assert len(await database.documents.search([1, 0, 0], 10)) == 2
            except BaseException:
                replacement.cancel()
                await asyncio.gather(replacement, return_exceptions=True)
                raise
        await asyncio.wait_for(replacement, timeout=5)
    assert (await database.corpus_info())["manifest"] == "new"
    assert len(await database.documents.search([1, 0, 0], 10)) == 1


async def test_schema_constraints_and_sqlalchemy_dependency_error(database, settings):
    with pytest.raises(IntegrityError):
        async with database.engine.begin() as conn:
            await conn.execute(
                insert(CorpusRow).values(singleton=False, manifest="bad", profile={})
            )
    chat_id = await database.chats.create()
    with pytest.raises(IntegrityError):
        async with database.engine.begin() as conn:
            await conn.execute(
                insert(TurnRow).values(id=uuid4(), chat_id=chat_id, question="?", status="invalid")
            )

    app = create_app(settings, database, object())

    @app.get("/test-db-error")
    async def database_error():
        raise OperationalError("test", {}, ConnectionError("unavailable"))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.get("/test-db-error")
    assert response.status_code == 503
    assert response.json()["code"] == "dependency_unavailable"
