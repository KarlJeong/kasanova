from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_hybrid() -> MagicMock:
    hybrid = MagicMock()
    hybrid.search = AsyncMock(
        return_value=[
            {
                "doc_id": "kasa_member",
                "content": "테이블명: kasa_member\n설명: ...",
                "source": "merged_schema.jsonl",
                "chunk_index": 0,
                "score": 0.9,
            },
            {
                "doc_id": "kasa_trading_order",
                "content": "테이블명: kasa_trading_order\n설명: ...",
                "source": "merged_schema.jsonl",
                "chunk_index": 0,
                "score": 0.8,
            },
        ]
    )
    return hybrid


class TestSchemaSearcher:
    async def test_returns_table_name_and_schema(
        self, mock_hybrid: MagicMock
    ) -> None:
        from app.rag.schema_searcher import SchemaSearcher

        searcher = SchemaSearcher(mock_hybrid)
        results = await searcher.search("회원 수 조회")

        assert len(results) == 2
        assert results[0] == {
            "table_name": "kasa_member",
            "schema": "테이블명: kasa_member\n설명: ...",
        }
        assert results[1]["table_name"] == "kasa_trading_order"

    async def test_calls_hybrid_with_top_k_10(
        self, mock_hybrid: MagicMock
    ) -> None:
        from app.rag.schema_searcher import SchemaSearcher

        searcher = SchemaSearcher(mock_hybrid)
        await searcher.search("질문")

        mock_hybrid.search.assert_awaited_once_with(
            "질문", top_k=10
        )

    async def test_empty_results(
        self, mock_hybrid: MagicMock
    ) -> None:
        from app.rag.schema_searcher import SchemaSearcher

        mock_hybrid.search = AsyncMock(return_value=[])
        searcher = SchemaSearcher(mock_hybrid)
        results = await searcher.search("질문")
        assert results == []
