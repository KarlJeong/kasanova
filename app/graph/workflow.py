import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.graph.nodes import call_llm
from app.graph.state import KasaNovaState
from app.graph.tools import create_web_search_tool
from app.rag.retrieval_tool import create_retrieval_tool
from app.rag.searcher import HybridSearcher

logger = logging.getLogger(__name__)


def build_workflow(
    checkpointer: BaseCheckpointSaver,
    llm: BaseChatModel,
    searcher: HybridSearcher,
) -> StateGraph:
    web_search = create_web_search_tool()
    retrieval_tool = create_retrieval_tool(searcher)
    tools = [web_search, retrieval_tool]
    llm_with_tools = llm.bind_tools(tools)
    llm_base = llm
    tool_node = ToolNode(tools)

    async def _call_llm(state: KasaNovaState) -> dict:
        return await call_llm(
            state, llm_with_tools, llm_base
        )

    async def _tool_node(state: KasaNovaState) -> dict:
        logger.info("[ToolNode] 도구 실행 시작")
        result = await tool_node.ainvoke(state)
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage):
                content = str(msg.content)
                logger.info(
                    "[ToolNode] %s 실행 완료 (결과 %d자)",
                    msg.name,
                    len(content),
                )
        logger.info("[ToolNode] → call_llm 재호출")
        return result

    builder = StateGraph(KasaNovaState)
    builder.add_node("call_llm", _call_llm)
    builder.add_node("tools", _tool_node)

    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges("call_llm", tools_condition)
    builder.add_edge("tools", "call_llm")

    return builder.compile(checkpointer=checkpointer)
