from langchain_core.messages import SystemMessage

from faq_service.application.models.agent_models import ModerationResult
from faq_service.application.models.agent_state import AgentState
from faq_service.domain.errors import ServiceError


def create_moderation_agent(llm, settings):
    prompts = {"moderation": (settings.prompts_path / "moderation.txt").read_text()}
    moderation_llm = llm.with_structured_output(
        ModerationResult, method=settings.structured_output_method
    )

    async def moderation(state: AgentState):
        try:
            output = await moderation_llm.ainvoke(
                [SystemMessage(content=prompts["moderation"]), *state["messages"]]
            )
            return {"moderation": ModerationResult.model_validate(output)}
        except ValueError as exc:
            raise ServiceError(
                "invalid_moderation", "Модель вернула некорректный результат модерации.", 502
            ) from exc

    return moderation
