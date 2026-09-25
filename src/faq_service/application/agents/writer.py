import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from faq_service.application.models.agent_models import WriterResult
from faq_service.application.models.agent_state import AgentState
from faq_service.domain.errors import ServiceError


def validate_answer(result: WriterResult, state: AgentState):
    if not result.answer.strip():
        raise ValueError("Answer must not be empty")
    decision = state["moderation"].decision
    if decision == "reject" and result.status != "rejected":
        raise ValueError("Moderation rejected this question")
    if decision == "clarify" and result.status != "needs_clarification":
        raise ValueError("Moderation requires clarification")
    if result.status == "answered" and not result.citations:
        raise ValueError("A factual answer needs citations from retrieved evidence")
    if result.status != "answered" and result.citations:
        raise ValueError("Only an answered question can have citations")
    evidence = {chunk.id: chunk for chunk in state["evidence"]}
    for citation in result.citations:
        chunk = evidence.get(citation.chunk_id)
        if not chunk or citation.document_id != chunk.document_id:
            raise ValueError("Citation points outside retrieved evidence")
        if not citation.quote.strip() or citation.quote not in chunk.text:
            raise ValueError("Citation quote must be an exact substring of its source")


def create_writer_agent(llm, settings):
    prompts = {n: (settings.prompts_path / f"{n}.txt").read_text() for n in ("writer", "repair")}
    writer_llm = llm.with_structured_output(WriterResult, method=settings.structured_output_method)

    async def writer(state: AgentState):
        # Previous dialogue is context, never evidence for new product facts.
        dialogue = []
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                dialogue.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage) and not message.tool_calls:
                dialogue.append({"role": "assistant", "content": message.content})
        payload = {
            "question": state["question"],
            "dialogue_context_not_evidence": dialogue,
            "moderation": state["moderation"].model_dump(),
            "evidence": [c.model_dump() for c in state["evidence"]],
        }
        messages = [
            SystemMessage(content=prompts["writer"]),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
        for attempt in range(2):
            try:
                output = WriterResult.model_validate(await writer_llm.ainvoke(messages))
                validate_answer(output, state)
                return {"result": output}
            except ValueError as exc:
                if attempt:
                    raise ServiceError(
                        "invalid_answer",
                        "Модель не смогла сформировать проверяемый ответ.",
                        502,
                    ) from exc
                messages.append(HumanMessage(content=prompts["repair"] + "\n" + str(exc)))

    return writer
