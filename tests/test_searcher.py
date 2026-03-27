from unittest.mock import AsyncMock, MagicMock

import pytest

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


class TestHybridSearch:
    @pytest.fixture
    def searcher(self) -> HybridSearcher:
        return HybridSearcher(
            os_client=_make_mock_os_client(
                [_make_hit(chunk_index=i) for i in range(5)]
            ),
            embedder=_make_mock_embedder(),
            index_name="test_index",
        )

    async def test_returns_top_k_results(
        self, searcher: HybridSearcher
    ) -> None:
        results = await searcher.search("테스트 질의")

        assert len(results) == 5

    async def test_result_contains_required_fields(
        self, searcher: HybridSearcher
    ) -> None:
        results = await searcher.search("테스트 질의")
        result = results[0]

        assert "content" in result
        assert "doc_id" in result
        assert "source" in result
        assert "chunk_index" in result
        assert "score" in result

    async def test_result_field_values(
        self, searcher: HybridSearcher
    ) -> None:
        results = await searcher.search("테스트 질의")
        result = results[0]

        assert result["content"] == "테스트 내용"
        assert result["doc_id"] == "report_abc12345"
        assert result["source"] == "report.pdf"
        assert result["chunk_index"] == 0
        assert result["score"] == 0.95

    async def test_calls_embedder_with_query(
        self, searcher: HybridSearcher
    ) -> None:
        await searcher.search("임베딩 테스트")

        searcher.embedder.encode.assert_called_once_with(
            ["임베딩 테스트"]
        )

    async def test_sends_hybrid_query(
        self, searcher: HybridSearcher
    ) -> None:
        await searcher.search("하이브리드 검색")

        searcher.os_client.search.assert_called_once()
        call_kwargs = (
            searcher.os_client.search.call_args[1]
        )
        assert call_kwargs["index"] == "test_index"

        body = call_kwargs["body"]
        hybrid = body["query"]["hybrid"]
        queries = hybrid["queries"]

        assert queries[0] == {
            "match": {"content": "하이브리드 검색"}
        }
        assert queries[1]["knn"]["embedding"]["vector"] == (
            [0.1] * 1024
        )
        assert queries[1]["knn"]["embedding"]["k"] == 20

    async def test_empty_index_returns_empty_list(
        self,
    ) -> None:
        searcher = HybridSearcher(
            os_client=_make_mock_os_client(hits=[]),
            embedder=_make_mock_embedder(),
            index_name="test_index",
        )

        results = await searcher.search("빈 인덱스")

        assert results == []

    async def test_custom_top_k(self) -> None:
        hits = [_make_hit(chunk_index=i) for i in range(3)]
        searcher = HybridSearcher(
            os_client=_make_mock_os_client(hits=hits),
            embedder=_make_mock_embedder(),
            index_name="test_index",
        )

        results = await searcher.search("질의", top_k=3)

        call_kwargs = (
            searcher.os_client.search.call_args[1]
        )
        assert call_kwargs["body"]["size"] == 3
        assert len(results) == 3
