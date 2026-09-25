from langchain_core.messages import AIMessage, SystemMessage

from faq_service.application.models.agent_state import AgentState
from faq_service.domain.errors import ServiceError


def create_rag_agent(llm, tools, settings):
    prompts = {"rag": (settings.prompts_path / "rag.txt").read_text()}
    rag_llm = llm.bind_tools(tools)

    async def rag(state: AgentState):
        output = await rag_llm.ainvoke(
            [
                SystemMessage(content=prompts["rag"]),
                *state["messages"],
            ]
        )
        if not isinstance(output, AIMessage) or output.invalid_tool_calls:
            raise ServiceError(
                "invalid_tool_call", "Модель вернула некорректный вызов поиска.", 502
            )
        # Execute at most one call per step, also for models ignoring parallel_tool_calls.
        calls = output.tool_calls[:1]
        if any(call["name"] != "vector_search" for call in calls):
            raise ServiceError("invalid_tool_call", "Модель запросила неизвестный инструмент.", 502)
        message = AIMessage(content=output.content, tool_calls=calls)
        return {"messages": [message], "tool_calls": state["tool_calls"] + len(calls)}

    return rag
