import json
import os
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from psycopg import sql
from psycopg.conninfo import make_conninfo

from faq_service.application.models.agent_models import ModerationResult, WriterResult
from faq_service.infrastructure.db.database import Database
from faq_service.settings.app_settings import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        openrouter_api_key="test",
        embedding_dimension=3,
        train_data_path=tmp_path / "train.jsonl",
        max_tool_calls=2,
        langfuse_enabled=False,
    )


class FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    async def aembed_documents(self, texts):
        self.calls += 1
        return [[1.0, 0.0, 0.0] if "выплат" in t.lower() else [0.0, 1.0, 0.0] for t in texts]

    async def aembed_query(self, text):
        return (await self.aembed_documents([text]))[0]


class FakeLLM:
    def __init__(self, decision="allow", repeat=False, bad_citation=False, skip_search=False):
        self.decision = decision
        self.repeat = repeat
        self.bad_citation = bad_citation
        self.skip_search = skip_search
        self.rag_calls = 0
        self.writer_calls = 0

    def with_structured_output(self, schema, **kwargs):
        async def respond(messages):
            if schema is ModerationResult:
                return ModerationResult(decision=self.decision, reason="test")
            self.writer_calls += 1
            payload = json.loads(messages[1].content)
            chunks = payload["evidence"]
            if self.decision != "allow":
                return WriterResult(
                    status="rejected" if self.decision == "reject" else "needs_clarification",
                    answer="Уточните вопрос.",
                    citations=[],
                )
            if not chunks:
                return WriterResult(status="no_answer", answer="Нет сведений.", citations=[])
            chunk = chunks[0]
            return WriterResult(
                status="answered",
                answer=chunk["text"] + " [1]",
                citations=[
                    {
                        "document_id": chunk["document_id"],
                        "chunk_id": "invented" if self.bad_citation else chunk["id"],
                        "quote": chunk["text"][:30],
                    }
                ],
            )

        return RunnableLambda(respond)

    def bind_tools(self, tools):
        async def respond(messages):
            self.rag_calls += 1
            if self.skip_search or (
                not self.repeat and any(isinstance(m, ToolMessage) for m in messages)
            ):
                return AIMessage(content="Поиск завершён.")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "vector_search",
                        "args": {"query": "Как повторить выплату?"},
                        "id": str(uuid4()),
                        "type": "tool_call",
                    }
                ],
            )

        return RunnableLambda(respond)


@pytest.fixture
def embeddings():
    return FakeEmbeddings()


@pytest.fixture
def dataset(settings):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Как повторить выплату?"},
                {"role": "assistant", "content": "Откройте выплату и нажмите Повторить."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Как связаться с поддержкой?"},
                {"role": "assistant", "content": "Напишите в чат поддержки."},
            ]
        },
    ]
    settings.train_data_path.write_text("\n".join(json.dumps(r) for r in rows))
    return settings.train_data_path


@pytest_asyncio.fixture
async def database(settings):
    dsn = os.getenv("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set TEST_DATABASE_URL to run real pgvector integration tests")
    schema = "test_" + uuid4().hex
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    db = Database(settings, make_conninfo(dsn, options=f"-c search_path={schema},public"))
    await db.initialize()
    try:
        yield db
    finally:
        async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
            await conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
