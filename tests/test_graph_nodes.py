from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.nodes import call_llm
from app.graph.state import KasaNovaState


class TestCallLlmNode:
    async def test_returns_ai_message(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="테스트 답변")

        state: KasaNovaState = {
            "messages": [HumanMessage(content="안녕하세요")]
        }
        result = await call_llm(state, mock_llm)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "테스트 답변"

    async def test_prepends_system_prompt(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="답변")

        state: KasaNovaState = {
            "messages": [HumanMessage(content="질문")]
        }
        await call_llm(state, mock_llm)

        call_args = mock_llm.ainvoke.call_args[0][0]
        assert call_args[0].type == "system"
        assert "KasaNova" in call_args[0].content

    async def test_preserves_message_history(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="답변2")

        state: KasaNovaState = {
            "messages": [
                HumanMessage(content="첫 질문"),
                AIMessage(content="첫 답변"),
                HumanMessage(content="두번째 질문"),
            ]
        }
        await call_llm(state, mock_llm)

        call_args = mock_llm.ainvoke.call_args[0][0]
        # SystemMessage + 3 history messages
        assert len(call_args) == 4

    async def test_propagates_llm_error(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = ConnectionError("서버 연결 실패")

        state: KasaNovaState = {
            "messages": [HumanMessage(content="질문")]
        }
        with pytest.raises(ConnectionError):
            await call_llm(state, mock_llm)
