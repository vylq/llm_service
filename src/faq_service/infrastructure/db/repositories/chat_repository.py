import hashlib
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from faq_service.domain.errors import ServiceError
from faq_service.infrastructure.db.locks import CORPUS_LOCK, advisory_lock
from faq_service.infrastructure.db.models import ChatRow, TurnRow


class ChatRepository:
    def __init__(self, db):
        self.db = db
        self.settings = db.settings

    async def create(self):
        chat_id = uuid4()
        async with self.db.connection() as conn:
            async with conn.begin():
                await conn.execute(insert(ChatRow).values(id=chat_id))
        return chat_id

    @asynccontextmanager
    async def session(self, chat_id: UUID):
        key = int.from_bytes(hashlib.sha256(chat_id.bytes).digest()[:8], signed=True)
        async with self.db.connection() as conn:
            async with conn.begin():
                await self._check_exists(conn, chat_id)
            async with advisory_lock(
                conn, key, ServiceError("chat_busy", "Дождитесь ответа на предыдущий запрос.", 409)
            ):
                async with advisory_lock(
                    conn,
                    CORPUS_LOCK,
                    ServiceError("index_updating", "Обновляется база знаний. Повторите запрос."),
                    shared=True,
                ):
                    # Under the chat lock, any pending turn belongs to an interrupted request.
                    async with conn.begin():
                        await conn.execute(
                            update(TurnRow)
                            .where(TurnRow.chat_id == chat_id, TurnRow.status == "pending")
                            .values(status="failed", error_code="interrupted")
                        )
                    yield conn

    async def _check_exists(self, conn: AsyncConnection, chat_id: UUID):
        if await conn.scalar(select(ChatRow.id).where(ChatRow.id == chat_id)) is None:
            raise ServiceError("chat_not_found", "Чат не найден.", 404)

    async def history(self, chat_id: UUID, limit: int = 50, offset: int = 0):
        async with self.db.connection() as conn:
            async with conn.begin():
                await self._check_exists(conn, chat_id)
                result = await conn.execute(
                    select(TurnRow)
                    .where(TurnRow.chat_id == chat_id)
                    .order_by(TurnRow.created_at, TurnRow.id)
                    .limit(limit)
                    .offset(offset)
                )
                return [dict(row) for row in result.mappings()]

    async def recent_history(self, conn: AsyncConnection, chat_id: UUID):
        async with conn.begin():
            result = await conn.execute(
                select(TurnRow.question, TurnRow.response)
                .where(TurnRow.chat_id == chat_id, TurnRow.status == "completed")
                .order_by(TurnRow.created_at.desc(), TurnRow.id.desc())
                .limit(self.settings.history_turns)
            )
            return [dict(row) for row in result.mappings()]

    async def begin_turn(self, conn: AsyncConnection, chat_id: UUID, question: str):
        run_id = uuid4()
        async with conn.begin():
            await conn.execute(
                insert(TurnRow).values(
                    id=run_id, chat_id=chat_id, question=question, status="pending"
                )
            )
        return run_id

    async def finish_turn(self, conn: AsyncConnection, run_id: UUID, response):
        async with conn.begin():
            await conn.execute(
                update(TurnRow)
                .where(TurnRow.id == run_id)
                .values(status="completed", response=response.model_dump(mode="json"))
            )

    async def fail_turn(self, conn: AsyncConnection, run_id: UUID, code: str):
        async with conn.begin():
            await conn.execute(
                update(TurnRow).where(TurnRow.id == run_id).values(status="failed", error_code=code)
            )
