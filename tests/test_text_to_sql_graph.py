"""text_to_sql LangGraph subgraph 노드별 단위 테스트.

각 노드는 deps와 state만 받는 작은 함수이므로, 통합 테스트와 별개로
노드 하나만 골라 검증할 수 있다. 이게 그래프 분해의 핵심 이득.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.rag.text_to_sql_graph import (
    TextToSqlDeps,
    _make_execute_sql_node,
    _make_format_result_node,
    _make_generate_sql_node,
    _make_retrieve_context_node,
    _route_after_generate,
    _route_after_retrieve,
    build_text_to_sql_graph,
)


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_schema_searcher() -> MagicMock:
    s = MagicMock()
    s.search = AsyncMock(
        return_value=[
            {
                "table_name": "kasa_member",
                "schema": (
                    "테이블명: kasa_member\n컬럼: id, joined_at"
                ),
            }
        ]
    )
    s.fetch_schemas_by_names = AsyncMock(return_value=[])
    return s


@pytest.fixture
def mock_kb_searcher() -> MagicMock:
    s = MagicMock()
    s.fetch_best_docs = AsyncMock(
        return_value=[
            {
                "doc_id": "offering_subscription_process",
                "content": (
                    "관련 테이블: kasa_offering,"
                    " kasa_offering_subscription"
                ),
                "source": "offering_subscription_process.md",
                "score": 0.82,
                "chunk_count": 3,
            }
        ]
    )
    return s


@pytest.fixture
def mock_mysql_client() -> MagicMock:
    c = MagicMock()
    c.execute_select = AsyncMock(
        return_value=[{"count": 42}]
    )
    return c


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="SELECT COUNT(*) FROM kasa_member"
        )
    )
    return llm


@pytest.fixture
def deps(
    mock_schema_searcher,
    mock_kb_searcher,
    mock_mysql_client,
    mock_llm,
) -> TextToSqlDeps:
    return TextToSqlDeps(
        schema_searcher=mock_schema_searcher,
        kb_searcher=mock_kb_searcher,
        mysql_client=mock_mysql_client,
        llm=mock_llm,
    )


# ────────────────────────────────────────────────────────────────────
# retrieve_context_node
# ────────────────────────────────────────────────────────────────────


class TestRetrieveContextNode:
    async def test_returns_schemas_and_kb_docs(
        self, deps
    ) -> None:
        node = _make_retrieve_context_node(deps)
        result = await node(
            {"query": "회원 수", "literals": []}
        )
        assert "schemas" in result
        assert "kb_docs" in result
        assert len(result["schemas"]) >= 1
        assert len(result["kb_docs"]) == 1

    async def test_kb_augments_missing_tables(
        self, deps, mock_schema_searcher
    ) -> None:
        # KB가 언급한 kasa_offering, kasa_offering_subscription 중
        # 스키마에 없는 것을 fetch_schemas_by_names로 끌어와야 함
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
            ]
        )
        node = _make_retrieve_context_node(deps)
        result = await node(
            {"query": "질문", "literals": []}
        )
        names = {s["table_name"] for s in result["schemas"]}
        assert "kasa_offering" in names
        assert "kasa_offering_subscription" in names
        mock_schema_searcher.fetch_schemas_by_names.assert_awaited_once()

    async def test_no_schemas_sets_error(
        self, deps, mock_schema_searcher
    ) -> None:
        mock_schema_searcher.search = AsyncMock(return_value=[])
        node = _make_retrieve_context_node(deps)
        result = await node(
            {"query": "질문", "literals": []}
        )
        assert result.get("error") is not None
        assert result["schemas"] == []

    async def test_kb_failure_proceeds_without_kb(
        self, deps, mock_kb_searcher
    ) -> None:
        mock_kb_searcher.fetch_best_docs = AsyncMock(
            side_effect=RuntimeError("kb down")
        )
        node = _make_retrieve_context_node(deps)
        result = await node(
            {"query": "질문", "literals": []}
        )
        assert result.get("error") is None
        assert result.get("kb_docs") == []
        assert len(result["schemas"]) >= 1

    async def test_schema_failure_sets_error(
        self, deps, mock_schema_searcher
    ) -> None:
        mock_schema_searcher.search = AsyncMock(
            side_effect=RuntimeError("os down")
        )
        node = _make_retrieve_context_node(deps)
        result = await node(
            {"query": "질문", "literals": []}
        )
        assert result.get("error") is not None


# ────────────────────────────────────────────────────────────────────
# generate_sql_node
# ────────────────────────────────────────────────────────────────────


class TestGenerateSqlNode:
    async def test_happy_path(self, deps) -> None:
        node = _make_generate_sql_node(deps)
        result = await node(
            {
                "query": "회원 수",
                "literals": [],
                "schemas": [
                    {
                        "table_name": "kasa_member",
                        "schema": "...",
                    }
                ],
                "kb_docs": [],
            }
        )
        assert result.get("sql") is not None
        assert result["sql"].startswith("SELECT")
        assert result.get("error") is None

    async def test_unknown_response_sets_error(
        self, deps, mock_llm
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="UNKNOWN: 컬럼이 없음"
            )
        )
        node = _make_generate_sql_node(deps)
        result = await node(
            {
                "query": "질문",
                "literals": [],
                "schemas": [
                    {"table_name": "x", "schema": "..."}
                ],
                "kb_docs": [],
            }
        )
        assert result.get("sql") is None
        assert result.get("error") is not None

    async def test_forbidden_keyword_blocked(
        self, deps, mock_llm
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="DROP TABLE kasa_member"
            )
        )
        node = _make_generate_sql_node(deps)
        result = await node(
            {
                "query": "질문",
                "literals": [],
                "schemas": [
                    {"table_name": "x", "schema": "..."}
                ],
                "kb_docs": [],
            }
        )
        assert result.get("sql") is None
        assert result.get("error") is not None

    async def test_strips_code_fence(
        self, deps, mock_llm
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="```sql\nSELECT 1\n```"
            )
        )
        node = _make_generate_sql_node(deps)
        result = await node(
            {
                "query": "질문",
                "literals": [],
                "schemas": [
                    {"table_name": "x", "schema": "..."}
                ],
                "kb_docs": [],
            }
        )
        assert result["sql"] == "SELECT 1"

    async def test_kb_top1_only_in_prompt(
        self, deps, mock_llm
    ) -> None:
        """본문 주입은 항상 top-1만. top-2는 프롬프트에 포함되지 않음."""
        node = _make_generate_sql_node(deps)
        await node(
            {
                "query": "질문",
                "literals": [],
                "schemas": [
                    {"table_name": "x", "schema": "..."}
                ],
                "kb_docs": [
                    {
                        "doc_id": "doc1",
                        "content": "TOP1_SENTINEL",
                        "source": "doc1.md",
                        "score": 0.8,
                        "chunk_count": 1,
                    },
                    {
                        "doc_id": "doc2",
                        "content": "TOP2_SENTINEL",
                        "source": "doc2.md",
                        "score": 0.7,
                        "chunk_count": 1,
                    },
                ],
            }
        )
        prompt = mock_llm.ainvoke.call_args.args[0][-1][1]
        assert "TOP1_SENTINEL" in prompt
        assert "TOP2_SENTINEL" not in prompt


# ────────────────────────────────────────────────────────────────────
# execute_sql_node
# ────────────────────────────────────────────────────────────────────


class TestExecuteSqlNode:
    async def test_happy_path(self, deps) -> None:
        node = _make_execute_sql_node(deps)
        result = await node({"sql": "SELECT 1"})
        assert result.get("rows") == [{"count": 42}]
        assert result.get("error") is None

    async def test_mysql_returns_none_sets_error(
        self, deps, mock_mysql_client
    ) -> None:
        mock_mysql_client.execute_select = AsyncMock(
            return_value=None
        )
        node = _make_execute_sql_node(deps)
        result = await node({"sql": "SELECT 1"})
        assert result.get("error") is not None

    async def test_missing_sql_sets_error(self, deps) -> None:
        node = _make_execute_sql_node(deps)
        result = await node({"sql": None})
        assert result.get("error") is not None


# ────────────────────────────────────────────────────────────────────
# format_result_node
# ────────────────────────────────────────────────────────────────────


class TestFormatResultNode:
    async def test_error_short_circuits(self) -> None:
        node = _make_format_result_node()
        result = await node({"error": "조회할 수 없습니다"})
        assert result["final_result"] == "조회할 수 없습니다"

    async def test_zero_rows(self) -> None:
        node = _make_format_result_node()
        result = await node({"rows": []})
        assert result["final_result"] == "조회된 데이터가 없습니다"

    async def test_none_rows(self) -> None:
        node = _make_format_result_node()
        result = await node({"rows": None})
        assert result["final_result"] == "조회할 수 없습니다"

    async def test_single_scalar(self) -> None:
        node = _make_format_result_node()
        result = await node({"rows": [{"count": 42}]})
        assert result["final_result"] == "결과 1건: 42"

    async def test_multi_row(self) -> None:
        node = _make_format_result_node()
        result = await node(
            {
                "rows": [
                    {"id": 1, "name": "a"},
                    {"id": 2, "name": "b"},
                ]
            }
        )
        assert "결과 2건" in result["final_result"]
        assert '"name": "a"' in result["final_result"]


# ────────────────────────────────────────────────────────────────────
# Routing
# ────────────────────────────────────────────────────────────────────


class TestRouting:
    def test_route_after_retrieve_to_generate(self) -> None:
        assert (
            _route_after_retrieve({"schemas": [{"x": 1}]})
            == "generate_sql"
        )

    def test_route_after_retrieve_to_format_on_error(
        self,
    ) -> None:
        assert (
            _route_after_retrieve({"error": "x"})
            == "format_result"
        )

    def test_route_after_generate_to_execute(self) -> None:
        assert (
            _route_after_generate({"sql": "SELECT 1"})
            == "execute_sql"
        )

    def test_route_after_generate_to_format_on_error(
        self,
    ) -> None:
        assert (
            _route_after_generate({"error": "x"})
            == "format_result"
        )


# ────────────────────────────────────────────────────────────────────
# Compiled graph end-to-end
# ────────────────────────────────────────────────────────────────────


class TestCompiledGraph:
    async def test_end_to_end_happy(self, deps) -> None:
        graph = build_text_to_sql_graph(deps)
        final = await graph.ainvoke(
            {"query": "회원 수", "literals": []}
        )
        assert final.get("final_result") == "결과 1건: 42"

    async def test_end_to_end_unknown(
        self, deps, mock_llm
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="UNKNOWN")
        )
        graph = build_text_to_sql_graph(deps)
        final = await graph.ainvoke(
            {"query": "질문", "literals": []}
        )
        assert (
            final.get("final_result") == "조회할 수 없습니다"
        )

    async def test_end_to_end_zero_rows(
        self, deps, mock_mysql_client
    ) -> None:
        mock_mysql_client.execute_select = AsyncMock(
            return_value=[]
        )
        graph = build_text_to_sql_graph(deps)
        final = await graph.ainvoke(
            {"query": "질문", "literals": []}
        )
        assert (
            final.get("final_result")
            == "조회된 데이터가 없습니다"
        )
