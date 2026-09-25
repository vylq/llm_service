import hashlib
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb

from faq_service.domain.errors import ServiceError
from faq_service.infrastructure.db.locks import CORPUS_LOCK


class ChatRepository:
    def __init__(self, db):
        self.db = db
        self.settings = db.settings

    async def create(self):
        chat_id = uuid4()
        async with self.db.connection() as conn:
            await conn.execute("INSERT INTO chats(id) VALUES (%s)", (chat_id,))
        return chat_id

    @asynccontextmanager
    async def session(self, chat_id: UUID):
        # Session-scoped locks release automatically if the process/connection dies.
        key = int.from_bytes(hashlib.sha256(chat_id.bytes).digest()[:8], signed=True)
        async with self.db.connection() as conn:
            exists = await (
                await conn.execute(
                    "SELECT id FROM chats WHERE id = %s",
                    (chat_id,),
                )
            ).fetchone()
            if not exists:
                raise ServiceError("chat_not_found", "Чат не найден.", 404)
            locked = await (
                await conn.execute(
                    "SELECT pg_try_advisory_lock(%s) AS ok",
                    (key,),
                )
            ).fetchone()
            if not locked["ok"]:
                raise ServiceError("chat_busy", "Дождитесь ответа на предыдущий запрос.", 409)
            shared = await (
                await conn.execute(
                    "SELECT pg_try_advisory_lock_shared(%s, %s) AS ok",
                    CORPUS_LOCK,
                )
            ).fetchone()
            if not shared["ok"]:
                raise ServiceError("index_updating", "Обновляется база знаний. Повторите запрос.")
            # No other request can be active in this chat now: pending means interrupted.
            await conn.execute(
                "UPDATE turns SET status='failed', error_code='interrupted' "
                "WHERE chat_id=%s AND status='pending'",
                (chat_id,),
            )
            yield conn

    async def history(self, chat_id: UUID, limit: int = 50, offset: int = 0):
        async with self.db.connection() as conn:
            exists = await (
                await conn.execute(
                    "SELECT id FROM chats WHERE id=%s",
                    (chat_id,),
                )
            ).fetchone()
            if not exists:
                raise ServiceError("chat_not_found", "Чат не найден.", 404)
            return await (
                await conn.execute(
                    "SELECT * FROM turns WHERE chat_id=%s "
                    "ORDER BY created_at, id LIMIT %s OFFSET %s",
                    (chat_id, limit, offset),
                )
            ).fetchall()

    async def recent_history(self, conn, chat_id: UUID):
        return await (
            await conn.execute(
                "SELECT question, response FROM turns WHERE chat_id=%s AND status='completed' "
                "ORDER BY created_at DESC, id DESC LIMIT %s",
                (chat_id, self.settings.history_turns),
            )
        ).fetchall()

    async def begin_turn(self, conn, chat_id: UUID, question: str):
        run_id = uuid4()
        await conn.execute(
            "INSERT INTO turns(id, chat_id, question, status) VALUES (%s, %s, %s, 'pending')",
            (run_id, chat_id, question),
        )
        return run_id

    async def finish_turn(self, conn, run_id: UUID, response):
        await conn.execute(
            "UPDATE turns SET status='completed', response=%s WHERE id=%s",
            (Jsonb(response.model_dump(mode="json")), run_id),
        )

    async def fail_turn(self, conn, run_id: UUID, code: str):
        await conn.execute(
            "UPDATE turns SET status='failed', error_code=%s WHERE id=%s",
            (code, run_id),
        )
