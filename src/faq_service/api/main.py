import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from faq_service.api.errors import register_handlers
from faq_service.api.routers import router
from faq_service.application.services.chat_service import ChatService
from faq_service.application.workflows.graph import build_graph
from faq_service.infrastructure.db.database import Database
from faq_service.infrastructure.llm.factory import chat_model, embedding_model
from faq_service.settings.app_settings import Settings


def create_app(settings=None, db=None, graph=None):
    @asynccontextmanager
    async def lifespan(app):
        config = settings or Settings()
        logging.basicConfig(level=config.log_level)
        database = db or Database(config)
        await database.initialize()
        workflow = graph or build_graph(
            config,
            database,
            chat_model(config),
            embedding_model(config),
        )
        service = ChatService(config, database, workflow)
        app.state.settings = config
        app.state.db = database
        app.state.service = service
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title="FAQ RAG service", version="0.1.0", lifespan=lifespan)

    register_handlers(app)
    app.include_router(router)
    return app


app = create_app()
