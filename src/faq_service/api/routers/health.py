from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health/live")
async def live():
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(request: Request):
    await request.app.state.db.check_ready()
    return {"status": "ready"}
