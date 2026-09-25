import json

from faq_service.domain.entities.document import Document, RetrievedChunk


class DocumentRepository:
    def __init__(self, db):
        self.db = db
        self.settings = db.settings

    async def search(self, vector, limit: int):
        async with self.db.connection() as conn:
            rows = await (
                await conn.execute(
                    """SELECT c.id, c.document_id, d.title, c.text,
                          c.embedding <=> %s::vector AS distance
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   ORDER BY c.embedding <=> %s::vector, c.id LIMIT %s""",
                    (json.dumps(vector), json.dumps(vector), limit),
                )
            ).fetchall()
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
            rows = await (
                await conn.execute(
                    "SELECT id, title, body, sources FROM documents WHERE id = ANY(%s)",
                    (ids,),
                )
            ).fetchall()
        by_id = {row["id"]: Document.model_validate(row) for row in rows}
        return [by_id[id_] for id_ in dict.fromkeys(ids) if id_ in by_id]
