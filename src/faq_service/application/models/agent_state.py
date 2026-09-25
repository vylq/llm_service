from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from faq_service.application.models.agent_models import ModerationResult, SearchResult, WriterResult
from faq_service.domain.entities.document import RetrievedChunk


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    moderation: ModerationResult | None
    searches: list[SearchResult]
    evidence: list[RetrievedChunk]
    tool_calls: int
    result: WriterResult | None
