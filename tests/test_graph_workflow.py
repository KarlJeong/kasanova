from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver

from app.graph.workflow import build_workflow


class TestBuildWorkflow:
    def test_returns_compiled_graph(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        checkpointer = MemorySaver()
        workflow = build_workflow(checkpointer, mock_llm)
        assert hasattr(workflow, "ainvoke")

    async def test_ainvoke_produces_ai_message(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="답변입니다")
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
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
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
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


class TestReActWorkflow:
    @patch("app.graph.tools.TavilyClient")
    @patch("app.graph.tools.get_settings")
    async def test_tool_node_executes_on_tool_calls(
        self, mock_settings: MagicMock, mock_tavily_cls: MagicMock
    ) -> None:
        mock_settings.return_value.TAVILY_API_KEY = "tvly-test"
        mock_tavily_cls.return_value.search.return_value = {
            "results": [{"title": "Result", "url": "https://ex.com"}]
        }

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "web_search",
                        "args": {"query": "test"},
                    }
                ],
            ),
            AIMessage(content="검색 결과 기반 답변"),
        ]
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        checkpointer = MemorySaver()
        workflow = build_workflow(checkpointer, mock_llm)

        config = {"configurable": {"thread_id": "react-test-1"}}
        result = await workflow.ainvoke(
            {"messages": [HumanMessage(content="검색해줘")]},
            config=config,
        )

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert last_msg.content == "검색 결과 기반 답변"
        # LLM이 2번 호출됨 (첫번째: tool_calls, 두번째: 최종 답변)
        assert mock_llm.ainvoke.call_count == 2

    @patch("app.graph.tools.TavilyClient")
    @patch("app.graph.tools.get_settings")
    async def test_tool_message_accumulated_in_state(
        self, mock_settings: MagicMock, mock_tavily_cls: MagicMock
    ) -> None:
        mock_settings.return_value.TAVILY_API_KEY = "tvly-test"
        mock_tavily_cls.return_value.search.return_value = {
            "results": [{"title": "Result"}]
        }

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "web_search",
                        "args": {"query": "test"},
                    }
                ],
            ),
            AIMessage(content="최종 답변"),
        ]
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        checkpointer = MemorySaver()
        workflow = build_workflow(checkpointer, mock_llm)

        config = {"configurable": {"thread_id": "react-test-2"}}
        result = await workflow.ainvoke(
            {"messages": [HumanMessage(content="검색")]},
            config=config,
        )

        # messages에 ToolMessage가 포함되어야 함
        tool_messages = [
            m
            for m in result["messages"]
            if isinstance(m, ToolMessage)
        ]
        assert len(tool_messages) >= 1

    @patch("app.graph.tools.TavilyClient")
    @patch("app.graph.tools.get_settings")
    async def test_no_tool_calls_goes_directly_to_end(
        self, mock_settings: MagicMock, mock_tavily_cls: MagicMock
    ) -> None:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(
            content="도구 없이 직접 답변"
        )
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        checkpointer = MemorySaver()
        workflow = build_workflow(checkpointer, mock_llm)

        config = {"configurable": {"thread_id": "react-test-3"}}
        result = await workflow.ainvoke(
            {"messages": [HumanMessage(content="안녕")]},
            config=config,
        )

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert last_msg.content == "도구 없이 직접 답변"
        # LLM은 1번만 호출
        assert mock_llm.ainvoke.call_count == 1
        # ToolMessage 없음
        tool_messages = [
            m
            for m in result["messages"]
            if isinstance(m, ToolMessage)
        ]
        assert len(tool_messages) == 0
