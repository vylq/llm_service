from contextlib import asynccontextmanager

from psycopg.conninfo import conninfo_to_dict
from sqlalchemy import DDL, URL
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from sqlalchemy.pool import NullPool

from faq_service.domain.errors import ServiceError
from faq_service.infrastructure.db.models import Base
from faq_service.infrastructure.db.repositories.chat_repository import ChatRepository
from faq_service.infrastructure.db.repositories.corpus_repository import CorpusRepository
from faq_service.infrastructure.db.repositories.document_repository import DocumentRepository
from faq_service.settings.app_settings import Settings


class Database:
    def __init__(self, settings: Settings, dsn: str | None = None):
        self.settings = settings
        self.dsn = dsn
        url = URL.create(
            "postgresql+psycopg",
            host=settings.database_host,
            port=settings.database_port,
            database=settings.database_name,
            username=settings.database_user,
            password=settings.database_password.get_secret_value(),
        )
        connect_args: dict[str, object] = {"connect_timeout": settings.database_connect_timeout}
        if dsn:
            # Keep support for PostgreSQL URLs and libpq connection strings used by callers.
            url = URL.create("postgresql+psycopg")
            connect_args.update(conninfo_to_dict(dsn))
            connect_args["connect_timeout"] = settings.database_connect_timeout
        # Closing a connection ends its PostgreSQL session, including any advisory locks.
        self.engine = create_async_engine(url, connect_args=connect_args, poolclass=NullPool)
        self.documents = DocumentRepository(self)
        self.chats = ChatRepository(self)
        self.corpus = CorpusRepository(self)

    @asynccontextmanager
    async def connection(self):
        async with self.engine.connect() as conn:
            yield conn

    async def initialize(self):
        async with self.engine.begin() as conn:
            # PostgreSQL extensions have no built-in SQLAlchemy schema construct.
            await conn.execute(DDL("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

    async def close(self):
        await self.engine.dispose()

    async def corpus_info(self, conn: AsyncConnection | None = None):
        return await self.corpus.info(conn)

    async def check_ready(self, conn=None):
        info = await self.corpus_info(conn)
        if not info:
            raise ServiceError("index_missing", "База знаний ещё не загружена.")
        if info["profile"] != self.settings.embedding_profile:
            raise ServiceError("embedding_mismatch", "Нужно пересчитать индекс для этих настроек.")
