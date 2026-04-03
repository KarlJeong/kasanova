"""쿼리별 상세 로그(query_results) 및 점수 통계(score_stats) 검증."""

from unittest.mock import AsyncMock, MagicMock

import pytest

_REQUIRED_QUERY_FIELDS = {
    "query_id",
    "category",
    "query",
    "ground_truth_doc_id",
    "retrieved_rank",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_10",
    "retrieved_docs",
    "rank1_score",
    "answer_score",
    "score_gap",
}

_REQUIRED_STATS_FIELDS = {
    "rank1_score_mean",
    "rank1_score_std",
    "rank1_rank2_gap_mean",
    "rank1_rank2_gap_min",
    "miss_count",
    "miss_query_ids",
}


def _make_searcher(
    results_by_query: dict[str, list[dict]],
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


def _make_result(
    doc_id: str, score: float, content: str = "테스트 내용",
) -> dict:
    return {
        "doc_id": doc_id,
        "score": score,
        "content": content,
        "source": "test.pdf",
        "chunk_index": 0,
    }


# ── fixtures ─────────────────────��────────────────────


@pytest.fixture
def hit_at_1_dataset() -> list[dict]:
    """정답이 1위인 데이터셋."""
    return [
        {
            "question": "연차 신청 절차는?",
            "ground_truth_answer": "답변",
            "ground_truth_doc_ids": ["doc_001"],
            "category": "hr",
        },
    ]


@pytest.fixture
def hit_at_3_dataset() -> list[dict]:
    """정답이 3위인 데이터셋."""
    return [
        {
            "question": "복리후생 항목은?",
            "ground_truth_answer": "답변",
            "ground_truth_doc_ids": ["doc_target"],
            "category": "benefit",
        },
    ]


@pytest.fixture
def miss_dataset() -> list[dict]:
    """정답이 검색 결과에 없는 데이터셋."""
    return [
        {
            "question": "장애 대응 절차는?",
            "ground_truth_answer": "답변",
            "ground_truth_doc_ids": ["doc_missing"],
            "category": "ops",
        },
    ]


@pytest.fixture
def mixed_dataset() -> list[dict]:
    """hit + miss 혼합 데이터셋."""
    return [
        {
            "question": "연차 신청 절차는?",
            "ground_truth_answer": "답변",
            "ground_truth_doc_ids": ["doc_001"],
            "category": "hr",
        },
        {
            "question": "장애 대응 절차는?",
            "ground_truth_answer": "답변",
            "ground_truth_doc_ids": ["doc_missing"],
            "category": "ops",
        },
    ]


# ── query_results 필드 검증 ──────────────────────────


class TestQueryResultsFields:
    async def test_required_fields_present(
        self, hit_at_1_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        searcher = _make_searcher({
            "연차 신청 절차는?": [
                _make_result("doc_001", 0.95),
                _make_result("doc_002", 0.80),
            ],
        })

        result = await evaluate_retrieval(
            searcher, hit_at_1_dataset
        )

        assert "query_results" in result
        qr = result["query_results"][0]
        assert _REQUIRED_QUERY_FIELDS <= set(qr.keys())

    async def test_hit_at_rank_1(
        self, hit_at_1_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        searcher = _make_searcher({
            "연차 신청 절차는?": [
                _make_result("doc_001", 0.95),
                _make_result("doc_002", 0.80),
            ],
        })

        result = await evaluate_retrieval(
            searcher, hit_at_1_dataset
        )

        qr = result["query_results"][0]
        assert qr["retrieved_rank"] == 1
        assert qr["hit_at_1"] is True
        assert qr["hit_at_3"] is True
        assert qr["hit_at_5"] is True
        assert qr["hit_at_10"] is True
        assert qr["score_gap"] == 0

    async def test_hit_at_rank_3(
        self, hit_at_3_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        searcher = _make_searcher({
            "복리후생 항목은?": [
                _make_result("doc_A", 0.95),
                _make_result("doc_B", 0.90),
                _make_result("doc_target", 0.85),
            ],
        })

        result = await evaluate_retrieval(
            searcher, hit_at_3_dataset
        )

        qr = result["query_results"][0]
        assert qr["retrieved_rank"] == 3
        assert qr["hit_at_1"] is False
        assert qr["hit_at_3"] is True
        assert qr["rank1_score"] == pytest.approx(0.95)
        assert qr["answer_score"] == pytest.approx(0.85)
        assert qr["score_gap"] == pytest.approx(0.10)

    async def test_miss(
        self, miss_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        searcher = _make_searcher({
            "장애 대응 절차는?": [
                _make_result("doc_X", 0.90),
                _make_result("doc_Y", 0.80),
            ],
        })

        result = await evaluate_retrieval(
            searcher, miss_dataset
        )

        qr = result["query_results"][0]
        assert qr["retrieved_rank"] is None
        assert qr["hit_at_1"] is False
        assert qr["hit_at_10"] is False
        assert qr["answer_score"] is None
        assert qr["score_gap"] is None


class TestRetrievedDocs:
    async def test_retrieved_docs_fields(
        self, hit_at_1_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        searcher = _make_searcher({
            "연차 신청 절차는?": [
                _make_result("doc_001", 0.95, "내용입니다"),
            ],
        })

        result = await evaluate_retrieval(
            searcher, hit_at_1_dataset
        )

        doc = result["query_results"][0]["retrieved_docs"][0]
        assert doc["rank"] == 1
        assert doc["doc_id"] == "doc_001"
        assert doc["score"] == pytest.approx(0.95)
        assert isinstance(doc["snippet"], str)

    async def test_snippet_max_100_chars(
        self, hit_at_1_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        long_content = "가" * 200
        searcher = _make_searcher({
            "연차 신청 절차는?": [
                _make_result("doc_001", 0.95, long_content),
            ],
        })

        result = await evaluate_retrieval(
            searcher, hit_at_1_dataset
        )

        snippet = result["query_results"][0]["retrieved_docs"][0]["snippet"]
        assert len(snippet) <= 100


# ── score_stats 검증 ─────────────────────────────────


class TestScoreStats:
    async def test_stats_structure(
        self, mixed_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        searcher = _make_searcher({
            "연차 신청 절차는?": [
                _make_result("doc_001", 0.95),
                _make_result("doc_002", 0.80),
            ],
            "장애 대응 절차는?": [
                _make_result("doc_X", 0.90),
                _make_result("doc_Y", 0.80),
            ],
        })

        result = await evaluate_retrieval(
            searcher, mixed_dataset
        )

        assert "score_stats" in result
        stats = result["score_stats"]
        assert "overall" in stats
        assert "by_category" in stats
        assert _REQUIRED_STATS_FIELDS <= set(
            stats["overall"].keys()
        )

    async def test_miss_count_and_ids(
        self, mixed_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        searcher = _make_searcher({
            "연차 신청 절차는?": [
                _make_result("doc_001", 0.95),
            ],
            "장애 대응 절차는?": [
                _make_result("doc_X", 0.90),
            ],
        })

        result = await evaluate_retrieval(
            searcher, mixed_dataset
        )

        stats = result["score_stats"]["overall"]
        assert stats["miss_count"] == 1
        assert "q_1" in stats["miss_query_ids"]

    async def test_category_stats(
        self, mixed_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        searcher = _make_searcher({
            "연차 신청 절차는?": [
                _make_result("doc_001", 0.95),
                _make_result("doc_002", 0.80),
            ],
            "장애 대응 절차는?": [
                _make_result("doc_X", 0.90),
                _make_result("doc_Y", 0.70),
            ],
        })

        result = await evaluate_retrieval(
            searcher, mixed_dataset
        )

        by_cat = result["score_stats"]["by_category"]
        assert "hr" in by_cat
        assert "ops" in by_cat
        assert by_cat["hr"]["miss_count"] == 0
        assert by_cat["ops"]["miss_count"] == 1


# ── 레그레션: 기존 classic_ir 구조 유지 ──────────────


class TestRegressionClassicIR:
    async def test_overall_and_by_category_unchanged(
        self, hit_at_1_dataset: list[dict],
    ) -> None:
        from eval.evaluate_retrieval import (
            evaluate_retrieval,
        )

        searcher = _make_searcher({
            "연차 신청 절차는?": [
                _make_result("doc_001", 0.95),
            ],
        })

        result = await evaluate_retrieval(
            searcher, hit_at_1_dataset
        )

        assert "overall" in result
        assert "by_category" in result
        assert "precision@1" in result["overall"]
        assert "mrr@1" in result["overall"]
        assert "ndcg@1" in result["overall"]
