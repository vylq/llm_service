from uuid import UUID

from faq_service.application.models.agent_models import WriterResult
from faq_service.domain.entities.document import Document


class ChatResponse(WriterResult):
    chat_id: UUID
    run_id: UUID
    documents: list[Document]
