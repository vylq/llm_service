from fastapi import APIRouter, Request

from faq_service.domain.errors import ServiceError

router = APIRouter()


@router.get("/api/v1/documents/{document_id}")
async def document(document_id: str, request: Request):
    found = await request.app.state.db.documents.get_many([document_id])
    if not found:
        raise ServiceError("document_not_found", "Документ не найден.", 404)
    return found[0]
