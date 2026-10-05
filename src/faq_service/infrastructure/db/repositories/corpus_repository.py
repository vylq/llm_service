from collections.abc import Iterable
from contextlib import asynccontextmanager
from itertools import batched

from sqlalchemy import delete, func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection

from faq_service.domain.entities.document import Document
from faq_service.infrastructure.db.locks import CORPUS_LOCK, INGEST_LOCK, advisory_lock
from faq_service.infrastructure.db.models import ChunkRow, CorpusRow, DocumentRow


class CorpusRepository:
    def __init__(self, db):
        self.db = db

    @asynccontextmanager
    async def ingestion(self):
        async with self.db.connection() as conn:
            async with advisory_lock(conn, INGEST_LOCK, ValueError("Another ingestion is running")):
                yield conn

    async def info(self, conn: AsyncConnection | None = None):
        if conn is None:
            async with self.db.connection() as own:
                return await self.info(own)
        async with conn.begin():
            result = await conn.execute(select(CorpusRow.manifest, CorpusRow.profile))
            row = result.mappings().first()
            return dict(row) if row is not None else None

    async def replace(
        self,
        conn: AsyncConnection,
        documents: list[Document],
        chunks: Iterable[dict],
        manifest: str,
        profile: dict,
        batch_size: int,
    ):
        """Atomically replace the index and its metadata after embeddings are ready."""
        async with conn.begin():
            await conn.execute(select(func.pg_advisory_xact_lock(*CORPUS_LOCK)))
            await conn.execute(delete(DocumentRow))
            for batch in batched(documents, batch_size):
                await conn.execute(
                    insert(DocumentRow),
                    [document.model_dump(mode="json") for document in batch],
                )
            for batch in batched(chunks, batch_size):
                await conn.execute(insert(ChunkRow), list(batch))
            statement = pg_insert(CorpusRow).values(
                singleton=True, manifest=manifest, profile=profile
            )
            await conn.execute(
                statement.on_conflict_do_update(
                    index_elements=[CorpusRow.singleton],
                    set_={
                        "manifest": statement.excluded.manifest,
                        "profile": statement.excluded.profile,
                    },
                )
            )
