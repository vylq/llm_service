import asyncio
import logging

from langchain_core.messages import AIMessage, HumanMessage

from faq_service.application.models.chat import ChatResponse
from faq_service.domain.errors import ServiceError
from faq_service.infrastructure.langfuse.tracing import Tracing

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, settings, db, graph):
        self.settings = settings
        self.db = db
        self.graph = graph
        self.tracing = Tracing(settings)

    async def answer(self, chat_id, question):
        async with self.db.chats.session(chat_id) as conn:
            run_id = await self.db.chats.begin_turn(conn, chat_id, question)
            try:
                async with asyncio.timeout(self.settings.request_timeout_seconds):
                    await self.db.check_ready(conn)
                    history = await self.db.chats.recent_history(conn, chat_id)
                    selected, total_chars = [], 0
                    for row in history:  # Newest first, keep complete turns only.
                        answer = row["response"]["answer"]
                        total_chars += len(row["question"]) + len(answer)
                        if total_chars > self.settings.history_chars:
                            break
                        selected.append((row["question"], answer))
                    messages = []
                    for q, a in reversed(selected):
                        messages.extend([HumanMessage(content=q), AIMessage(content=a)])
                    messages.append(HumanMessage(content=question))
                    config = {
                        "recursion_limit": self.settings.max_tool_calls * 3 + 6,
                        "metadata": {
                            "chat_id": str(chat_id),
                            "run_id": str(run_id),
                            "langfuse_session_id": str(chat_id),
                        },
                    }
                    config["callbacks"] = self.tracing.callbacks()
                    state = await self.graph.ainvoke(
                        {
                            "messages": messages,
                            "question": question,
                            "moderation": None,
                            "searches": [],
                            "evidence": [],
                            "tool_calls": 0,
                            "result": None,
                        },
                        config=config,
                    )
                    result = state["result"]
                    documents = await self.db.documents.get_many(
                        [chunk.document_id for chunk in state["evidence"]],
                    )
                    response = ChatResponse(
                        **result.model_dump(),
                        chat_id=chat_id,
                        run_id=run_id,
                        documents=documents,
                    )
                    await self.db.chats.finish_turn(conn, run_id, response)
                    return response
            except BaseException as exc:
                code = exc.code if isinstance(exc, ServiceError) else "request_failed"
                try:
                    await self.db.chats.fail_turn(conn, run_id, code)
                except Exception:
                    logger.error("Could not persist failure for run_id=%s", run_id)
                raise

    async def close(self):
        await self.tracing.close()
