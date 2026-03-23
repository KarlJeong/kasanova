from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.graph.nodes import call_llm
from app.graph.state import KasaNovaState
from app.graph.tools import web_search


def build_workflow(
    checkpointer: BaseCheckpointSaver, llm: BaseChatModel
) -> StateGraph:
    tools = [web_search]
    llm_with_tools = llm.bind_tools(tools)

    async def _call_llm(state: KasaNovaState) -> dict:
        return await call_llm(state, llm_with_tools)

    builder = StateGraph(KasaNovaState)
    builder.add_node("call_llm", _call_llm)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges("call_llm", tools_condition)
    builder.add_edge("tools", "call_llm")

    return builder.compile(checkpointer=checkpointer)
