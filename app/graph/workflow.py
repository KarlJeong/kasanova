from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import call_llm
from app.graph.state import KasaNovaState


def build_workflow(
    checkpointer: BaseCheckpointSaver, llm: BaseChatModel
) -> StateGraph:
    async def _call_llm(state: KasaNovaState) -> dict:
        return await call_llm(state, llm)

    builder = StateGraph(KasaNovaState)
    builder.add_node("call_llm", _call_llm)
    builder.add_edge(START, "call_llm")
    builder.add_edge("call_llm", END)
    return builder.compile(checkpointer=checkpointer)
