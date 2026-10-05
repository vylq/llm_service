"""Database schema; repositories execute Core statements over these mapped tables."""

from datetime import datetime
from uuid import UUID

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Text, func, true
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CorpusRow(Base):
    __tablename__ = "corpus"

    singleton: Mapped[bool] = mapped_column(primary_key=True, server_default=true())
    manifest: Mapped[str] = mapped_column(Text)
    profile: Mapped[dict] = mapped_column(JSONB)

    __table_args__ = (CheckConstraint(singleton.column, name="corpus_singleton_check"),)


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    sources: Mapped[list[dict]] = mapped_column(JSONB)


class ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(Text, ForeignKey("documents.id", ondelete="CASCADE"))
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(VECTOR())

    __table_args__ = (Index("chunks_document_idx", document_id),)


class ChatRow(Base):
    __tablename__ = "chats"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TurnRow(Base):
    __tablename__ = "turns"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    chat_id: Mapped[UUID] = mapped_column(ForeignKey("chats.id"))
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    response: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    error_code: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            status.column.in_(("pending", "completed", "failed")), name="turns_status_check"
        ),
        Index("turns_chat_idx", chat_id, created_at, id),
    )
