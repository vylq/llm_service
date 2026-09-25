from typing import Literal

from pydantic import BaseModel, ConfigDict

from faq_service.domain.entities.document import Citation, RetrievedChunk


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModerationResult(StrictModel):
    decision: Literal["allow", "clarify", "reject"]
    reason: str


class WriterResult(StrictModel):
    status: Literal["answered", "needs_clarification", "no_answer", "rejected"]
    answer: str
    citations: list[Citation]


class SearchResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
