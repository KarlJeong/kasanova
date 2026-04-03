"""Classic IR 평가 — ranx 기반 Precision, Recall, MAP, MRR, NDCG at k."""

import logging
import statistics
from typing import Any

from ranx import Qrels, Run, evaluate

from app.rag.searcher import HybridSearcher

logger = logging.getLogger(__name__)

_K_VALUES = [1, 3, 5, 10]
_METRICS = ["precision", "recall", "map", "mrr", "ndcg"]


def _build_metric_list(
    k_values: list[int],
) -> list[str]:
    """ranx evaluate에 전달할 메트릭 문자열 목록을 생성한다."""
    return [
        f"{m}@{k}" for k in k_values for m in _METRICS
    ]


def _evaluate_subset(
    qrels_dict: dict[str, dict[str, int]],
    run_dict: dict[str, dict[str, float]],
    metrics: list[str],
) -> dict[str, float]:
    """Qrels/Run 부분집합에 대해 ranx evaluate를 실행한다."""
    if not qrels_dict or not run_dict:
        return {m: 0.0 for m in metrics}

    # ranx는 빈 run entry를 허용하지 않으므로 필터링
    filtered_run = {
        qid: docs
        for qid, docs in run_dict.items()
        if docs
    }
    if not filtered_run:
        return {m: 0.0 for m in metrics}

    # run에 남은 쿼리만 qrels에서도 사용
    filtered_qrels = {
        qid: rels
        for qid, rels in qrels_dict.items()
        if qid in filtered_run
    }
    if not filtered_qrels:
        return {m: 0.0 for m in metrics}

    qrels = Qrels(filtered_qrels)
    run = Run(filtered_run)
    result = evaluate(qrels, run, metrics)
    return dict(result)


def _build_query_result(
    query_id: str,
    item: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """단일 쿼리의 상세 결과를 구성한다."""
    gt_id = item["ground_truth_doc_ids"][0]

    # retrieved_docs 구성 (중복 제거, 상위 10개)
    seen: set[str] = set()
    retrieved_docs: list[dict[str, Any]] = []
    for r in results:
        doc_id = r["doc_id"]
        if doc_id not in seen:
            seen.add(doc_id)
            retrieved_docs.append({
                "rank": len(retrieved_docs) + 1,
                "doc_id": doc_id,
                "score": float(r["score"]),
                "snippet": r.get("content", "")[:100],
            })

    # retrieved_rank 계산
    retrieved_rank: int | None = None
    answer_score: float | None = None
    for doc in retrieved_docs:
        if doc["doc_id"] == gt_id:
            retrieved_rank = doc["rank"]
            answer_score = doc["score"]
            break

    rank1_score = (
        retrieved_docs[0]["score"] if retrieved_docs
        else None
    )

    if answer_score is not None and rank1_score is not None:
        score_gap = round(rank1_score - answer_score, 10)
    else:
        score_gap = None

    return {
        "query_id": query_id,
        "category": item["category"],
        "query": item["question"],
        "ground_truth_doc_id": gt_id,
        "retrieved_rank": retrieved_rank,
        "hit_at_1": retrieved_rank is not None
            and retrieved_rank <= 1,
        "hit_at_3": retrieved_rank is not None
            and retrieved_rank <= 3,
        "hit_at_5": retrieved_rank is not None
            and retrieved_rank <= 5,
        "hit_at_10": retrieved_rank is not None
            and retrieved_rank <= 10,
        "retrieved_docs": retrieved_docs,
        "rank1_score": rank1_score,
        "answer_score": answer_score,
        "score_gap": score_gap,
    }


def _safe_stdev(values: list[float]) -> float:
    """표본이 1개 이하일 때 0.0을 반환하는 표준편차."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _compute_stats_for_group(
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """쿼리 결과 그룹에 대해 score_stats를 산출한다."""
    rank1_scores: list[float] = []
    rank1_rank2_gaps: list[float] = []
    miss_ids: list[str] = []

    for qr in items:
        docs = qr["retrieved_docs"]
        if docs:
            rank1_scores.append(docs[0]["score"])
        if len(docs) >= 2:
            rank1_rank2_gaps.append(
                docs[0]["score"] - docs[1]["score"]
            )

        if not qr["hit_at_1"]:
            miss_ids.append(qr["query_id"])

    return {
        "rank1_score_mean": (
            statistics.mean(rank1_scores)
            if rank1_scores else 0.0
        ),
        "rank1_score_std": _safe_stdev(rank1_scores),
        "rank1_rank2_gap_mean": (
            statistics.mean(rank1_rank2_gaps)
            if rank1_rank2_gaps else 0.0
        ),
        "rank1_rank2_gap_min": (
            min(rank1_rank2_gaps)
            if rank1_rank2_gaps else 0.0
        ),
        "miss_count": len(miss_ids),
        "miss_query_ids": miss_ids,
    }


def _compute_score_stats(
    query_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """전체 및 카테고리별 score_stats를 산출한다."""
    overall = _compute_stats_for_group(query_results)

    by_category: dict[str, dict[str, Any]] = {}
    categories = {"hr", "ops", "benefit"}
    for cat in categories:
        cat_items = [
            qr for qr in query_results
            if qr["category"] == cat
        ]
        if cat_items:
            by_category[cat] = _compute_stats_for_group(
                cat_items
            )

    return {
        "overall": overall,
        "by_category": by_category,
    }


async def evaluate_retrieval(
    searcher: HybridSearcher,
    dataset: list[dict[str, Any]],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """평가셋에 대해 IR 지표를 계산한다."""
    if k_values is None:
        k_values = _K_VALUES

    max_k = max(k_values)
    metrics = _build_metric_list(k_values)

    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}
    category_map: dict[str, str] = {}
    query_results: list[dict[str, Any]] = []

    for idx, item in enumerate(dataset):
        query_id = f"q_{idx}"
        category_map[query_id] = item["category"]

        # Ground truth
        qrels_dict[query_id] = {
            doc_id: 1
            for doc_id in item["ground_truth_doc_ids"]
        }

        # Retrieval
        results = await searcher.search(
            query=item["question"],
            top_k=max_k,
            category=item["category"],
        )

        # 쿼리별 상세 결과 수집
        qr = _build_query_result(query_id, item, results)
        query_results.append(qr)

        # ranx용 run_dict 구성 (중복 제거된 doc→score)
        run_dict[query_id] = {
            doc["doc_id"]: doc["score"]
            for doc in qr["retrieved_docs"]
        }

    # ranx 집계 지표
    overall = _evaluate_subset(
        qrels_dict, run_dict, metrics
    )

    categories = {"hr", "ops", "benefit"}
    by_category: dict[str, dict[str, float]] = {}
    for cat in categories:
        cat_qids = [
            qid
            for qid, c in category_map.items()
            if c == cat
        ]
        if not cat_qids:
            continue
        cat_qrels = {
            qid: qrels_dict[qid] for qid in cat_qids
        }
        cat_run = {
            qid: run_dict[qid] for qid in cat_qids
        }
        by_category[cat] = _evaluate_subset(
            cat_qrels, cat_run, metrics
        )

    # 점수 분포 통계
    score_stats = _compute_score_stats(query_results)

    return {
        "overall": overall,
        "by_category": by_category,
        "query_results": query_results,
        "score_stats": score_stats,
    }
