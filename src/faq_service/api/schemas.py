from uuid import UUID

from pydantic import BaseModel, Field

from faq_service.application.models.agent_models import StrictModel


class MessageRequest(StrictModel):
    message: str = Field(min_length=1, max_length=20000)


class ChatCreated(BaseModel):
    chat_id: UUID
