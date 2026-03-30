from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import BaseTool

from app.rag.retrieval_tool import create_retrieval_tool
from app.rag.searcher import HybridSearcher


def _make_mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.encode.return_value = [[0.1] * 1024]
    return embedder


def _make_mock_os_client(
    hits: list[dict] | None = None,
) -> AsyncMock:
    client = AsyncMock()
    client.search = AsyncMock(
        return_value={
            "hits": {
                "total": {"value": len(hits or [])},
                "hits": hits or [],
            }
        }
    )
    return client


def _make_hit(
    doc_id: str = "report_abc12345",
    chunk_index: int = 0,
    content: str = "테스트 내용",
    filename: str = "report.pdf",
    score: float = 0.95,
) -> dict:
    return {
        "_score": score,
        "_source": {
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "content": content,
            "metadata": {"filename": filename},
        },
    }


class TestRetrievalTool:
    @pytest.fixture
    def searcher(self) -> HybridSearcher:
        return HybridSearcher(
            os_client=_make_mock_os_client(
                [_make_hit(chunk_index=i) for i in range(10)]
            ),
            embedder=_make_mock_embedder(),
            index_name="test_index",
        )

    @pytest.fixture
    def tool(self, searcher: HybridSearcher) -> BaseTool:
        return create_retrieval_tool(searcher)

    def test_is_langchain_base_tool(
        self, tool: BaseTool
    ) -> None:
        assert isinstance(tool, BaseTool)

    async def test_returns_string_result(
        self, tool: BaseTool
    ) -> None:
        result = await tool.ainvoke({"query": "테스트 질의"})

        assert isinstance(result, str)

    async def test_calls_searcher_with_top_k_10(
        self, tool: BaseTool, searcher: HybridSearcher
    ) -> None:
        await tool.ainvoke({"query": "테스트 질의"})

        searcher.os_client.search.assert_called_once()
        call_kwargs = searcher.os_client.search.call_args[1]
        assert call_kwargs["body"]["size"] == 10

    async def test_result_contains_content_and_score(
        self, tool: BaseTool
    ) -> None:
        result = await tool.ainvoke({"query": "테스트 질의"})

        assert "테스트 내용" in result
        assert "0.95" in result

    async def test_result_contains_source_filename(
        self, tool: BaseTool
    ) -> None:
        result = await tool.ainvoke({"query": "테스트 질의"})

        assert "report.pdf" in result

    async def test_opensearch_failure_returns_error_string(
        self,
    ) -> None:
        failing_client = AsyncMock()
        failing_client.search = AsyncMock(
            side_effect=ConnectionError("연결 실패")
        )
        searcher = HybridSearcher(
            os_client=failing_client,
            embedder=_make_mock_embedder(),
            index_name="test_index",
        )
        tool = create_retrieval_tool(searcher)

        result = await tool.ainvoke({"query": "테스트"})

        assert isinstance(result, str)
        assert "오류" in result

    async def test_empty_results_returns_message(
        self,
    ) -> None:
        searcher = HybridSearcher(
            os_client=_make_mock_os_client(hits=[]),
            embedder=_make_mock_embedder(),
            index_name="test_index",
        )
        tool = create_retrieval_tool(searcher)

        result = await tool.ainvoke({"query": "없는 문서"})

        assert isinstance(result, str)
        assert "검색 결과가 없습니다" in result
