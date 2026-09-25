from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from faq_service.application.agents.moderation import create_moderation_agent
from faq_service.application.agents.rag import create_rag_agent
from faq_service.application.agents.writer import create_writer_agent
from faq_service.application.models.agent_state import AgentState
from faq_service.application.tools.vector_search import create_vector_search_tool
from faq_service.application.workflows.collect import create_collector


def build_graph(settings, db, llm, embeddings):
    vector_search = create_vector_search_tool(settings, db, embeddings)
    moderation = create_moderation_agent(llm, settings)
    rag = create_rag_agent(llm, [vector_search], settings)
    writer = create_writer_agent(llm, settings)
    collect = create_collector(settings)
    graph = StateGraph(AgentState)
    graph.add_node("moderation", moderation)
    graph.add_node("rag", rag)
    graph.add_node("tools", ToolNode([vector_search], handle_tool_errors=False))
    graph.add_node("collect", collect)
    graph.add_node("writer", writer)
    graph.add_edge(START, "moderation")
    graph.add_conditional_edges(
        "moderation",
        lambda s: "rag" if s["moderation"].decision == "allow" else "writer",
        {"rag": "rag", "writer": "writer"},
    )
    graph.add_conditional_edges(
        "rag",
        lambda s: "tools" if s["messages"][-1].tool_calls else "writer",
        {"tools": "tools", "writer": "writer"},
    )
    graph.add_edge("tools", "collect")
    graph.add_conditional_edges(
        "collect",
        lambda s: "rag" if s["tool_calls"] < settings.max_tool_calls else "writer",
        {"rag": "rag", "writer": "writer"},
    )
    graph.add_edge("writer", END)
    return graph.compile()
