"""Idempotent JSONL import; run with python -m faq_service.application.ingestion.pipeline."""

import argparse
import asyncio
import hashlib
import json
import tempfile
import unicodedata

from faq_service.domain.entities.document import Document, Source
from faq_service.domain.validation import validate_vector
from faq_service.infrastructure.db.database import Database
from faq_service.infrastructure.llm.factory import embedding_model
from faq_service.settings.app_settings import Settings

PLACEHOLDERS = {"!", "фейк", "card", "q", "1"}


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def load_documents(path):
    raw = path.read_bytes()
    file_hash = hashlib.sha256(raw).hexdigest()
    documents: dict[str, Document] = {}
    rejected = []
    for line, text in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        try:
            row = json.loads(text)
            messages = row["messages"]
            if [m["role"] for m in messages] != ["user", "assistant"]:
                raise ValueError("Expected user/assistant pair")
            q, a = [m["content"] for m in messages]
            if not all(isinstance(v, str) and v.strip() for v in [q, a]):
                raise ValueError("Empty or non-string content")
            if normalize(a).casefold() in PLACEHOLDERS:
                raise ValueError("placeholder")
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"line": line, "reason": str(exc)})
            continue
        id_ = digest([normalize(q), normalize(a)])
        source = Source(file=path.name, line=line, file_sha256=file_hash)
        if id_ in documents:
            documents[id_].sources.append(source)
        else:
            documents[id_] = Document(id=id_, title=q, body=a, sources=[source])
    if not documents:
        raise ValueError("Dataset has no usable documents; existing index was not changed")
    return list(documents.values()), rejected, file_hash


def chunk_text(text: str, size: int, overlap: int):
    """Bounded character chunks, prefer sentence/word boundaries; preserve original text."""
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n", start + size // 2, end),
                text.rfind(". ", start + size // 2, end),
                text.rfind(" ", start + size // 2, end),
            )
            if boundary > start:
                end = boundary + 1
        yield text[start:end]
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def spooled_chunks(spool):
    for line in spool:
        id_, document_id, text, vector = json.loads(line)
        yield {"id": id_, "document_id": document_id, "text": text, "embedding": vector}


async def ingest(settings: Settings, db: Database, embeddings, rebuild: bool = False):
    documents, rejected, file_hash = load_documents(settings.train_data_path)
    manifest = digest([file_hash, settings.embedding_profile])
    await db.initialize()
    async with db.corpus.ingestion() as conn:
        previous = await db.corpus_info(conn)
        if previous and previous["manifest"] == manifest:
            return {"status": "unchanged", "documents": len(documents)}
        if previous and not rebuild:
            raise ValueError("Corpus/settings changed. Run ingestion with --rebuild explicitly.")
        chunks = [
            (digest([doc.id, index, text]), doc.id, text, f"Вопрос: {doc.title}\nОтвет: {text}")
            for doc in documents
            for index, text in enumerate(
                chunk_text(
                    doc.body,
                    settings.chunk_chars,
                    settings.chunk_overlap,
                )
            )
        ]
        # Spool vectors to disk; no database transaction is open during network calls.
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as spool:
            for start in range(0, len(chunks), settings.embedding_batch_size):
                batch = chunks[start : start + settings.embedding_batch_size]
                vectors = await embeddings.aembed_documents([item[3] for item in batch])
                if len(vectors) != len(batch):
                    raise ValueError("Embedding API returned wrong number of vectors")
                for item, vector in zip(batch, vectors, strict=True):
                    validate_vector(vector, settings.embedding_dimension)
                    spool.write(json.dumps([*item[:3], vector], ensure_ascii=False) + "\n")
                print(f"Embedded {min(start + len(batch), len(chunks))}/{len(chunks)}", flush=True)
            spool.seek(0)
            await db.corpus.replace(
                conn,
                documents,
                spooled_chunks(spool),
                manifest,
                settings.embedding_profile,
                settings.embedding_batch_size,
            )
    return {
        "status": "imported",
        "documents": len(documents),
        "chunks": len(chunks),
        "quarantined": rejected,
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    db = Database(settings)
    try:
        result = await ingest(settings, db, embedding_model(settings), args.rebuild)
    finally:
        await db.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
