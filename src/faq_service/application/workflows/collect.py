from langchain_core.messages import ToolMessage

from faq_service.application.models.agent_models import SearchResult
from faq_service.application.models.agent_state import AgentState
from faq_service.domain.errors import ServiceError


def create_collector(settings):
    async def collect(state: AgentState):
        message = state["messages"][-1]
        if not isinstance(message, ToolMessage) or not message.artifact:
            raise ServiceError("search_failed", "Инструмент поиска не вернул результат.", 502)
        result = SearchResult.model_validate(message.artifact)
        best = {chunk.id: chunk for chunk in state["evidence"]}
        for chunk in result.chunks:
            if chunk.id not in best or chunk.distance < best[chunk.id].distance:
                best[chunk.id] = chunk
        evidence = sorted(best.values(), key=lambda c: (c.distance, c.id))
        return {
            "searches": [*state["searches"], result],
            "evidence": evidence[: settings.max_evidence_chunks],
        }

    return collect
