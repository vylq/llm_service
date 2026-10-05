from sqlalchemy import select

from faq_service.domain.entities.document import Document, RetrievedChunk
from faq_service.infrastructure.db.models import ChunkRow, DocumentRow


class DocumentRepository:
    def __init__(self, db):
        self.db = db
        self.settings = db.settings

    async def search(self, vector, limit: int):
        distance = ChunkRow.embedding.cosine_distance(vector)
        async with self.db.connection() as conn:
            async with conn.begin():
                result = await conn.execute(
                    select(
                        ChunkRow.id,
                        ChunkRow.document_id,
                        DocumentRow.title,
                        ChunkRow.text,
                        distance.label("distance"),
                    )
                    .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
                    .order_by(distance, ChunkRow.id)
                    .limit(limit)
                )
                rows = result.mappings().all()
        threshold = self.settings.max_cosine_distance
        return [
            RetrievedChunk.model_validate(row)
            for row in rows
            if threshold is None or row["distance"] <= threshold
        ]

    async def get_many(self, ids: list[str]):
        if not ids:
            return []
        async with self.db.connection() as conn:
            async with conn.begin():
                result = await conn.execute(select(DocumentRow).where(DocumentRow.id.in_(ids)))
                rows = result.mappings().all()
        by_id = {row["id"]: Document.model_validate(row) for row in rows}
        return [by_id[id_] for id_ in dict.fromkeys(ids) if id_ in by_id]
