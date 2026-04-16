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
    searcher.fetch_schemas_by_names = AsyncMock(
        return_value=[]
    )
    return searcher


@pytest.fixture
def mock_kb_searcher() -> MagicMock:
    searcher = MagicMock()
    searcher.fetch_best_docs = AsyncMock(
        return_value=[
            {
                "doc_id": "offering_subscription_process",
                "content": (
                    "관련 테이블: kasa_offering,"
                    " kasa_offering_subscription,"
                    " kasa_subscription_transaction"
                ),
                "source": "offering_subscription_process.md",
                "score": 0.82,
                "chunk_count": 3,
            }
        ]
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

    async def test_truncates_at_max_rows(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_mysql_client.execute_select = AsyncMock(
            return_value=[
                {"id": i} for i in range(1500)
            ]
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "목록"})
        assert "결과 1500건" in result
        assert "상위 1000건만 표시" in result

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
        mock_kb_searcher.fetch_best_docs.assert_awaited_once()
        call = mock_kb_searcher.fetch_best_docs.call_args
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
        human_content = llm_messages[-1][1]
        assert "## 도메인 참고" in human_content
        assert "kasa_offering_subscription" in human_content

    async def test_no_kb_match_proceeds_without_domain(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        mock_kb_searcher.fetch_best_docs = AsyncMock(
            return_value=[]
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        result = await tool.ainvoke({"query": "질문"})
        assert result == "결과 1건: 42"
        mock_schema_searcher.fetch_schemas_by_names.assert_not_awaited()

    async def test_kb_augments_schema_with_missing_tables(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        """KB 문서가 언급한 테이블 중 스키마 후보에 없는 것을
        fetch_schemas_by_names로 가져와 후보에 합쳐야 한다."""
        mock_schema_searcher.fetch_schemas_by_names = AsyncMock(
            return_value=[
                {
                    "table_name": "kasa_offering",
                    "schema": "테이블명: kasa_offering\n"
                    "컬럼: id, dabs_code",
                },
                {
                    "table_name": "kasa_offering_subscription",
                    "schema": "테이블명: kasa_offering_subscription\n"
                    "컬럼: id, member_id, offering_id",
                },
                {
                    "table_name": "kasa_subscription_transaction",
                    "schema": "테이블명: kasa_subscription_transaction\n"
                    "컬럼: id, subscription_id",
                },
            ]
        )

        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        await tool.ainvoke({"query": "질문"})

        mock_schema_searcher.fetch_schemas_by_names.assert_awaited_once()
        passed = (
            mock_schema_searcher.fetch_schemas_by_names.call_args.args[0]
        )
        assert "kasa_offering" in passed
        assert "kasa_offering_subscription" in passed
        assert "kasa_subscription_transaction" in passed
        # 이미 스키마 후보에 있는 테이블은 재조회하지 않음
        assert "kasa_member" not in passed

        human_content = mock_llm.ainvoke.call_args.args[0][-1][1]
        assert "kasa_offering" in human_content
        assert "dabs_code" in human_content

    async def test_kb_augmentation_skipped_when_all_present(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        """KB가 언급한 테이블이 이미 전부 스키마 후보에 있으면
        fetch_schemas_by_names를 호출하지 않는다."""
        mock_schema_searcher.search = AsyncMock(
            return_value=[
                {
                    "table_name": "kasa_offering",
                    "schema": "...",
                },
                {
                    "table_name": "kasa_offering_subscription",
                    "schema": "...",
                },
                {
                    "table_name": "kasa_subscription_transaction",
                    "schema": "...",
                },
            ]
        )
        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        await tool.ainvoke({"query": "질문"})
        mock_schema_searcher.fetch_schemas_by_names.assert_not_awaited()

    async def test_multi_doc_table_union(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        """멀티 도메인 쿼리에서 여러 KB 문서가 돌아오면 각 문서의
        테이블명을 합집합으로 수집해 스키마 보강에 사용한다."""
        mock_kb_searcher.fetch_best_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": "offering_subscription_process",
                    "content": (
                        "관련 테이블: kasa_offering,"
                        " kasa_offering_subscription"
                    ),
                    "source": "offering_subscription_process.md",
                    "score": 0.72,
                    "chunk_count": 5,
                },
                {
                    "doc_id": "dividend_process",
                    "content": (
                        "관련 테이블: kasa_dividend,"
                        " kasa_dividend_history"
                    ),
                    "source": "dividend_process.md",
                    "score": 0.65,
                    "chunk_count": 4,
                },
                {
                    "doc_id": "market_trading_process",
                    "content": (
                        "관련 테이블: kasa_trading_order"
                    ),
                    "source": "market_trading_process.md",
                    "score": 0.55,
                    "chunk_count": 3,
                },
            ]
        )
        mock_schema_searcher.fetch_schemas_by_names = AsyncMock(
            return_value=[
                {
                    "table_name": "kasa_offering",
                    "schema": "...",
                },
                {
                    "table_name": "kasa_offering_subscription",
                    "schema": "...",
                },
                {
                    "table_name": "kasa_dividend",
                    "schema": "...",
                },
                {
                    "table_name": "kasa_dividend_history",
                    "schema": "...",
                },
                {
                    "table_name": "kasa_trading_order",
                    "schema": "...",
                },
            ]
        )

        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        await tool.ainvoke({"query": "질문"})

        # 세 KB 문서 테이블 합집합이 보강 요청으로 전달돼야 함
        mock_schema_searcher.fetch_schemas_by_names.assert_awaited_once()
        passed = (
            mock_schema_searcher.fetch_schemas_by_names.call_args.args[0]
        )
        # 청약 도메인
        assert "kasa_offering" in passed
        assert "kasa_offering_subscription" in passed
        # 배당 도메인
        assert "kasa_dividend" in passed
        assert "kasa_dividend_history" in passed
        # 거래 도메인
        assert "kasa_trading_order" in passed
        # 중복 없음
        assert len(passed) == len(set(passed))

    async def test_multi_doc_only_top1_content_injected(
        self,
        mock_schema_searcher,
        mock_kb_searcher,
        mock_mysql_client,
        mock_llm,
    ) -> None:
        """KB 여러 문서가 돌아와도 본문 주입은 top-1만. top-2,
        top-3 문서 본문은 프롬프트에 포함되지 않아야 한다."""
        mock_kb_searcher.fetch_best_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": "offering_subscription_process",
                    "content": "SUBSCRIPTION_ONLY_SENTINEL",
                    "source": "offering_subscription_process.md",
                    "score": 0.72,
                    "chunk_count": 5,
                },
                {
                    "doc_id": "dividend_process",
                    "content": "DIVIDEND_ONLY_SENTINEL",
                    "source": "dividend_process.md",
                    "score": 0.65,
                    "chunk_count": 4,
                },
            ]
        )

        tool = _make_tool(
            mock_schema_searcher,
            mock_kb_searcher,
            mock_mysql_client,
            mock_llm,
        )
        await tool.ainvoke({"query": "질문"})

        human_content = mock_llm.ainvoke.call_args.args[0][-1][1]
        # top-1 본문은 주입됨
        assert "SUBSCRIPTION_ONLY_SENTINEL" in human_content
        # top-2 본문은 주입되지 않음 (프롬프트 비대화 방지)
        assert "DIVIDEND_ONLY_SENTINEL" not in human_content


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
        mock_kb_searcher.fetch_best_docs = AsyncMock(
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


class TestExtractKbTables:
    def test_extracts_comma_separated_list(self) -> None:
        from app.rag.text_to_sql_tool import _extract_kb_tables

        content = (
            "관련 테이블: kasa_offering, kasa_offering_subscription, "
            "kasa_subscription_transaction"
        )
        assert _extract_kb_tables(content) == [
            "kasa_offering",
            "kasa_offering_subscription",
            "kasa_subscription_transaction",
        ]

    def test_dedups_repeated_mentions(self) -> None:
        from app.rag.text_to_sql_tool import _extract_kb_tables

        content = (
            "kasa_member 회원 정보.\n"
            "가입 이력은 kasa_member 테이블에서 관리.\n"
            "kasa_offering 공모 정보."
        )
        assert _extract_kb_tables(content) == [
            "kasa_member",
            "kasa_offering",
        ]

    def test_ignores_non_kasa_tokens(self) -> None:
        from app.rag.text_to_sql_tool import _extract_kb_tables

        content = (
            "청약(subscription) 프로세스에서 kasa_offering이 생성된다.\n"
            "not_kasa_foo 는 무시."
        )
        assert _extract_kb_tables(content) == ["kasa_offering"]

    def test_empty_returns_empty(self) -> None:
        from app.rag.text_to_sql_tool import _extract_kb_tables

        assert _extract_kb_tables("") == []
