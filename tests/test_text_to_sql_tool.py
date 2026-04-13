from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def mock_schema_searcher() -> MagicMock:
    searcher = MagicMock()
    searcher.search = AsyncMock(
        return_value=[
            {
                "table_name": "kasa_member",
                "schema": "테이블명: kasa_member\n컬럼: id, joined_at",
            },
        ]
    )
    return searcher


@pytest.fixture
def mock_retrieval_tool() -> MagicMock:
    tool = MagicMock()
    tool.ainvoke = AsyncMock(
        return_value="[1] (score: 0.8) [kb.md] 회원 가입일 기준..."
    )
    return tool


@pytest.fixture
def mock_mysql_client() -> MagicMock:
    client = MagicMock()
    client.execute_select = AsyncMock(
        return_value=[{"count": 42}]
    )
    return client


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(
                content="SELECT COUNT(*) FROM kasa_member WHERE joined_at >= '2026-04-01'"
            ),
            AIMessage(content="이번 달 신규 가입 회원은 42명입니다."),
        ]
    )
    return llm


def _make_tool(
    mock_schema_searcher,
    mock_retrieval_tool,
    mock_mysql_client,
    mock_llm,
):
    from app.rag.text_to_sql_tool import (
        create_text_to_sql_tool,
    )

    return create_text_to_sql_tool(
        schema_searcher=mock_schema_searcher,
        retrieval_tool=mock_retrieval_tool,
        mysql_client=mock_mysql_client,
        llm=mock_llm,
    )


class TestHappyPath:
    async def test_returns_summary(
        self,
        mock_schema_searcher,
        mock_retrieval_tool,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        tool = _make_tool(
            mock_schema_searcher,
            mock_retrieval_tool,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke(
            {"query": "이번 달 신규 가입 회원 수는?"}
        )
        assert "42명" in result

    async def test_retrieval_tool_called_with_kb_category(
        self,
        mock_schema_searcher,
        mock_retrieval_tool,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        tool = _make_tool(
            mock_schema_searcher,
            mock_retrieval_tool,
            mock_mysql_client,
            mock_llm,
        )
        await tool.ainvoke({"query": "질문"})
        mock_retrieval_tool.ainvoke.assert_awaited_once()
        kwargs_or_arg = (
            mock_retrieval_tool.ainvoke.call_args.args[0]
        )
        assert kwargs_or_arg["category"] == "kb"
        assert kwargs_or_arg["query"] == "질문"


class TestErrorPaths:
    async def test_llm_returns_unknown(
        self,
        mock_schema_searcher,
        mock_retrieval_tool,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="UNKNOWN")
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_retrieval_tool,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert result == "조회할 수 없습니다"
        mock_mysql_client.execute_select.assert_not_awaited()

    async def test_forbidden_keyword_blocked(
        self,
        mock_schema_searcher,
        mock_retrieval_tool,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="DROP TABLE kasa_member"
            )
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_retrieval_tool,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert result == "조회할 수 없습니다"
        mock_mysql_client.execute_select.assert_not_awaited()

    async def test_non_select_blocked(
        self,
        mock_schema_searcher,
        mock_retrieval_tool,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="UPDATE kasa_member SET name='x'"
            )
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_retrieval_tool,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert result == "조회할 수 없습니다"

    async def test_mysql_failure(
        self,
        mock_schema_searcher,
        mock_retrieval_tool,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_mysql_client.execute_select = AsyncMock(
            return_value=None
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_retrieval_tool,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert result == "조회할 수 없습니다"

    async def test_zero_rows(
        self,
        mock_schema_searcher,
        mock_retrieval_tool,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_mysql_client.execute_select = AsyncMock(
            return_value=[]
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_retrieval_tool,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert result == "조회된 데이터가 없습니다"

    async def test_domain_knowledge_failure_proceeds(
        self,
        mock_schema_searcher,
        mock_retrieval_tool,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_retrieval_tool.ainvoke = AsyncMock(
            side_effect=RuntimeError("kb down")
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_retrieval_tool,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert "42명" in result

    async def test_strips_sql_code_fence(
        self,
        mock_schema_searcher,
        mock_retrieval_tool,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(
                    content="```sql\nSELECT 1\n```"
                ),
                AIMessage(content="결과: 1"),
            ]
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_retrieval_tool,
            mock_mysql_client,
            mock_llm,
        )
        await tool.ainvoke({"query": "질문"})
        executed = (
            mock_mysql_client.execute_select.call_args.args[0]
        )
        assert executed.strip().startswith("SELECT")
        assert "```" not in executed
