import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.eval.conftest import _make_mock_embedder


class TestHitRate:
    def test_hit_in_top_k(self) -> None:
        from eval.evaluate_retrieval import hit_rate

        assert hit_rate(["doc_A"], ["doc_A", "doc_B"], k=1) == 1.0

    def test_miss_in_top_k(self) -> None:
        from eval.evaluate_retrieval import hit_rate

        assert hit_rate(["doc_A"], ["doc_B", "doc_C"], k=2) == 0.0

    def test_hit_at_boundary(self) -> None:
        from eval.evaluate_retrieval import hit_rate

        assert (
            hit_rate(
                ["doc_A"], ["doc_B", "doc_C", "doc_A"], k=3
            )
            == 1.0
        )

    def test_hit_beyond_k(self) -> None:
        from eval.evaluate_retrieval import hit_rate

        assert (
            hit_rate(
                ["doc_A"], ["doc_B", "doc_C", "doc_A"], k=2
            )
            == 0.0
        )


class TestMRR:
    def test_first_position(self) -> None:
        from eval.evaluate_retrieval import mrr

        assert mrr(["doc_A"], ["doc_A", "doc_B"], k=5) == 1.0

    def test_third_position(self) -> None:
        from eval.evaluate_retrieval import mrr

        result = mrr(
            ["doc_A"],
            ["doc_B", "doc_C", "doc_A"],
            k=5,
        )
        assert result == pytest.approx(1 / 3)

    def test_not_found(self) -> None:
        from eval.evaluate_retrieval import mrr

        assert mrr(["doc_A"], ["doc_B", "doc_C"], k=5) == 0.0


class TestPrecisionAtK:
    def test_one_relevant_in_three(self) -> None:
        from eval.evaluate_retrieval import precision_at_k

        result = precision_at_k(
            ["doc_A"], ["doc_A", "doc_B", "doc_C"], k=3
        )
        assert result == pytest.approx(1 / 3)

    def test_all_relevant(self) -> None:
        from eval.evaluate_retrieval import precision_at_k

        result = precision_at_k(
            ["doc_A", "doc_B"], ["doc_A", "doc_B"], k=2
        )
        assert result == 1.0

    def test_none_relevant(self) -> None:
        from eval.evaluate_retrieval import precision_at_k

        result = precision_at_k(
            ["doc_A"], ["doc_B", "doc_C"], k=2
        )
        assert result == 0.0


class TestNDCGAtK:
    def test_relevant_at_position_0(self) -> None:
        from eval.evaluate_retrieval import ndcg_at_k

        result = ndcg_at_k(
            ["doc_A"], ["doc_A", "doc_B"], k=2
        )
        assert result == pytest.approx(1.0)

    def test_relevant_at_position_1(self) -> None:
        from eval.evaluate_retrieval import ndcg_at_k

        result = ndcg_at_k(
            ["doc_A"], ["doc_B", "doc_A"], k=2
        )
        expected = (1 / math.log2(3)) / 1.0
        assert result == pytest.approx(expected)

    def test_not_found(self) -> None:
        from eval.evaluate_retrieval import ndcg_at_k

        result = ndcg_at_k(
            ["doc_A"], ["doc_B", "doc_C"], k=2
        )
        assert result == 0.0


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
            assert f"hit_rate@{k}" in result["overall"]
            assert f"mrr@{k}" in result["overall"]
            assert f"precision@{k}" in result["overall"]
            assert f"ndcg@{k}" in result["overall"]

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
