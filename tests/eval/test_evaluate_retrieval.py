from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.eval.conftest import _make_mock_embedder

_ALL_METRICS = [
    "precision",
    "recall",
    "map",
    "mrr",
    "ndcg",
]


class TestEvaluateRetrieval:
    def _make_searcher(
        self, results_by_query: dict[str, list[dict]],
    ) -> MagicMock:
        from app.rag.searcher import HybridSearcher

        searcher = MagicMock(spec=HybridSearcher)

        async def _search(
            query: str,
            top_k: int = 5,
            category: str | None = None,
        ) -> list[dict]:
            return results_by_query.get(query, [])

        searcher.search = AsyncMock(side_effect=_search)
        return searcher

    async def test_returns_overall_and_by_category(
        self, sample_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        results_map = {
            "연차 신청 절차는?": [
                {"doc_id": "doc_001", "score": 0.9},
            ],
            "복리후생 항목은?": [
                {"doc_id": "doc_002", "score": 0.8},
            ],
            "서비스 장애 대응 절차는?": [
                {"doc_id": "doc_003", "score": 0.7},
            ],
        }
        searcher = self._make_searcher(results_map)

        result = await evaluate_retrieval(
            searcher, sample_dataset
        )

        assert "overall" in result
        assert "by_category" in result
        assert "hr" in result["by_category"]
        assert "ops" in result["by_category"]

    async def test_multiple_k_values(
        self, sample_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        results_map = {
            q["question"]: [
                {
                    "doc_id": q["ground_truth_doc_ids"][0],
                    "score": 0.9,
                }
            ]
            for q in sample_dataset
        }
        searcher = self._make_searcher(results_map)

        result = await evaluate_retrieval(
            searcher, sample_dataset
        )

        for k in [1, 3, 5, 10]:
            for m in _ALL_METRICS:
                assert (
                    f"{m}@{k}" in result["overall"]
                ), f"{m}@{k} missing from overall"

    async def test_perfect_retrieval(
        self, sample_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        results_map = {
            q["question"]: [
                {
                    "doc_id": q["ground_truth_doc_ids"][0],
                    "score": 1.0,
                }
            ]
            for q in sample_dataset
        }
        searcher = self._make_searcher(results_map)

        result = await evaluate_retrieval(
            searcher, sample_dataset
        )

        assert result["overall"]["precision@1"] == pytest.approx(1.0)
        assert result["overall"]["mrr@1"] == pytest.approx(1.0)
        assert result["overall"]["ndcg@1"] == pytest.approx(1.0)

    async def test_empty_results(
        self, sample_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        searcher = self._make_searcher({})
        searcher.search = AsyncMock(return_value=[])

        result = await evaluate_retrieval(
            searcher, sample_dataset
        )

        for m in _ALL_METRICS:
            assert (
                result["overall"][f"{m}@1"] == 0.0
            ), f"{m}@1 should be 0.0"

    async def test_category_filter_applied(
        self, sample_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        searcher = self._make_searcher({})
        searcher.search = AsyncMock(return_value=[])

        await evaluate_retrieval(
            searcher, sample_dataset
        )

        for call in searcher.search.call_args_list:
            kwargs = call[1] if call[1] else {}
            args = call[0] if call[0] else ()
            assert "category" in kwargs or len(args) >= 3
