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
def mock_kb_searcher() -> MagicMock:
    searcher = MagicMock()
    searcher.fetch_best_doc = AsyncMock(
        return_value={
            "doc_id": "offering_subscription_process",
            "content": "청약 = kasa_offering_subscription 사용",
            "source": "offering_subscription_process.md",
            "score": 0.82,
            "chunk_count": 3,
        }
    )
    return searcher


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
        return_value=AIMessage(
            content="SELECT COUNT(*) FROM kasa_member WHERE joined_at >= '2026-04-01'"
        )
    )
    return llm


def _make_tool(
    mock_schema_searcher,
    mock_kb_searcher,
    mock_mysql_client,
    mock_llm,
):
    from app.rag.text_to_sql_tool import (
        create_text_to_sql_tool,
    )

    return create_text_to_sql_tool(
        schema_searcher=mock_schema_searcher,
        kb_searcher=mock_kb_searcher,
        mysql_client=mock_mysql_client,
        llm=mock_llm,
    )


class TestHappyPath:
    async def test_single_scalar_result(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke(
            {"query": "이번 달 신규 가입 회원 수는?"}
        )
        assert result == "결과 1건: 42"
        # SQL must not leak into tool result
        assert "SELECT" not in result
        assert "```" not in result

    async def test_multi_row_result_is_json(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_mysql_client.execute_select = AsyncMock(
            return_value=[
                {"id": 1, "name": "a"},
                {"id": 2, "name": "b"},
            ]
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "목록"})
        assert result.startswith("결과 2건:")
        assert '"name": "a"' in result
        assert "SELECT" not in result

    async def test_truncates_at_10_rows(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_mysql_client.execute_select = AsyncMock(
            return_value=[
                {"id": i} for i in range(25)
            ]
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "목록"})
        assert "결과 25건" in result
        assert "상위 10건만 표시" in result

    async def test_kb_searcher_called_with_kb_category(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        await tool.ainvoke({"query": "질문"})
        mock_kb_searcher.fetch_best_doc.assert_awaited_once()
        call = mock_kb_searcher.fetch_best_doc.call_args
        # query는 positional 또는 keyword로 전달될 수 있음
        passed_query = (
            call.args[0] if call.args else call.kwargs["query"]
        )
        assert passed_query == "질문"
        assert call.kwargs.get("category") == "kb"

    async def test_kb_content_injected_into_prompt(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        await tool.ainvoke({"query": "질문"})
        llm_messages = mock_llm.ainvoke.call_args.args[0]
        # system + human
        human_content = llm_messages[-1][1]
        assert "kasa_offering_subscription 사용" in human_content
        assert "## 도메인 참고" in human_content

    async def test_no_kb_match_proceeds_without_domain(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_kb_searcher.fetch_best_doc = AsyncMock(
            return_value=None
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert result == "결과 1건: 42"
        human_content = mock_llm.ainvoke.call_args.args[0][-1][1]
        assert "(없음)" in human_content


class TestErrorPaths:
    async def test_llm_returns_unknown(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="UNKNOWN")
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert result == "조회할 수 없습니다"
        mock_mysql_client.execute_select.assert_not_awaited()

    async def test_forbidden_keyword_blocked(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
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
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert result == "조회할 수 없습니다"
        mock_mysql_client.execute_select.assert_not_awaited()

    async def test_non_select_blocked(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
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
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert result == "조회할 수 없습니다"

    async def test_mysql_failure(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_mysql_client.execute_select = AsyncMock(
            return_value=None
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert result == "조회할 수 없습니다"

    async def test_zero_rows(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_mysql_client.execute_select = AsyncMock(
            return_value=[]
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert result == "조회된 데이터가 없습니다"

    async def test_kb_failure_proceeds(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_kb_searcher.fetch_best_doc = AsyncMock(
            side_effect=RuntimeError("kb down")
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert result == "결과 1건: 42"

    async def test_strips_sql_code_fence(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="```sql\nSELECT 1\n```"
            )
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        await tool.ainvoke({"query": "질문"})
        executed = (
            mock_mysql_client.execute_select.call_args.args[0]
        )
        assert executed.strip().startswith("SELECT")
        assert "```" not in executed
