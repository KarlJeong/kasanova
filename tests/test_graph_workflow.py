from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.graph.workflow import build_workflow


class TestBuildWorkflow:
    def test_returns_compiled_graph(self) -> None:
        mock_llm = AsyncMock()
        checkpointer = MemorySaver()
        workflow = build_workflow(checkpointer, mock_llm)
        assert hasattr(workflow, "ainvoke")

    async def test_ainvoke_produces_ai_message(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="답변입니다")
        checkpointer = MemorySaver()
        workflow = build_workflow(checkpointer, mock_llm)

        config = {"configurable": {"thread_id": "test-thread-1"}}
        result = await workflow.ainvoke(
            {"messages": [HumanMessage(content="질문")]}, config=config
        )

        assert len(result["messages"]) >= 1
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert last_msg.content == "답변입니다"

    async def test_maintains_thread_history(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            AIMessage(content="첫 답변"),
            AIMessage(content="두번째 답변"),
        ]
        checkpointer = MemorySaver()
        workflow = build_workflow(checkpointer, mock_llm)

        config = {"configurable": {"thread_id": "test-thread-2"}}

        await workflow.ainvoke(
            {"messages": [HumanMessage(content="첫 질문")]},
            config=config,
        )

        result = await workflow.ainvoke(
            {"messages": [HumanMessage(content="두번째 질문")]},
            config=config,
        )

        # 두번째 호출 시 LLM에 전달된 메시지에 히스토리 포함 확인
        second_call_args = mock_llm.ainvoke.call_args_list[1][0][0]
        # SystemMessage + HumanMessage(첫) + AIMessage(첫) + HumanMessage(둘째)
        assert len(second_call_args) == 4
        assert second_call_args[1].content == "첫 질문"
        assert second_call_args[2].content == "첫 답변"
        assert second_call_args[3].content == "두번째 질문"
