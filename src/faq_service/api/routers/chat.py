from uuid import UUID

from fastapi import APIRouter, Query, Request

from faq_service.api.schemas import ChatCreated, MessageRequest
from faq_service.application.models.chat import ChatResponse
from faq_service.domain.errors import ServiceError

router = APIRouter()


@router.post("/api/v1/chats", response_model=ChatCreated, status_code=201)
async def create_chat(request: Request):
    return ChatCreated(chat_id=await request.app.state.db.chats.create())


@router.post("/api/v1/chats/{chat_id}/messages", response_model=ChatResponse)
async def send_message(chat_id: UUID, body: MessageRequest, request: Request):
    question = body.message.strip()
    if not question or len(question) > request.app.state.settings.max_input_chars:
        raise ServiceError("invalid_input", "Недопустимая длина сообщения.", 422)
    return await request.app.state.service.answer(chat_id, question)


@router.get("/api/v1/chats/{chat_id}/messages")
async def history(
    chat_id: UUID,
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    return {"turns": await request.app.state.db.chats.history(chat_id, limit, offset)}
