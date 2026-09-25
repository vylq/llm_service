from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from faq_service.domain.errors import ServiceError
from faq_service.infrastructure.db.repositories.chat_repository import ChatRepository
from faq_service.infrastructure.db.repositories.document_repository import DocumentRepository
from faq_service.settings.app_settings import Settings

# Two-key locks for corpus operations cannot collide with single-bigint chat locks.


class Database:
    def __init__(self, settings: Settings, dsn: str | None = None):
        self.settings = settings
        self.dsn = dsn
        self.documents = DocumentRepository(self)
        self.chats = ChatRepository(self)

    @asynccontextmanager
    async def connection(self):
        kwargs = (
            {}
            if self.dsn
            else {
                "host": self.settings.database_host,
                "port": self.settings.database_port,
                "dbname": self.settings.database_name,
                "user": self.settings.database_user,
                "password": self.settings.database_password.get_secret_value(),
            }
        )
        async with await psycopg.AsyncConnection.connect(
            self.dsn or "",
            **kwargs,
            autocommit=True,
            row_factory=dict_row,
            connect_timeout=self.settings.database_connect_timeout,
        ) as conn:
            yield conn

    async def initialize(self):
        async with self.connection() as conn:
            async with conn.transaction():
                await conn.execute(Path(__file__).with_name("schema.sql").read_text())

    async def corpus_info(self, conn=None):
        if conn is None:
            async with self.connection() as own:
                return await self.corpus_info(own)
        return await (await conn.execute("SELECT manifest, profile FROM corpus")).fetchone()

    async def check_ready(self, conn=None):
        info = await self.corpus_info(conn)
        if not info:
            raise ServiceError("index_missing", "База знаний ещё не загружена.")
        if info["profile"] != self.settings.embedding_profile:
            raise ServiceError("embedding_mismatch", "Нужно пересчитать индекс для этих настроек.")
