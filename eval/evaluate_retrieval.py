"""Classic IR 평가 — Hit Rate, MRR, Precision, NDCG at k."""

import logging
import math
from typing import Any

from app.rag.searcher import HybridSearcher

logger = logging.getLogger(__name__)

_K_VALUES = [1, 3, 5, 10]


def hit_rate(
    ground_truth_ids: list[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    """top-k 안에 정답 문서가 1개 이상 포함되면 1.0."""
    top_k = retrieved_ids[:k]
    gt_set = set(ground_truth_ids)
    return 1.0 if gt_set & set(top_k) else 0.0


def mrr(
    ground_truth_ids: list[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    """정답 문서가 처음 등장한 순위의 역수."""
    gt_set = set(ground_truth_ids)
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in gt_set:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(
    ground_truth_ids: list[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    """top-k 중 정답 문서 비율."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    gt_set = set(ground_truth_ids)
    relevant = sum(1 for d in top_k if d in gt_set)
    return relevant / len(top_k)


def ndcg_at_k(
    ground_truth_ids: list[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    """순위 가중치를 반영한 정규화 누적 이득."""
    gt_set = set(ground_truth_ids)
    top_k = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        if doc_id in gt_set:
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG
    n_relevant = min(len(gt_set), k)
    idcg = sum(
        1.0 / math.log2(i + 2) for i in range(n_relevant)
    )

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


async def evaluate_retrieval(
    searcher: HybridSearcher,
    dataset: list[dict[str, Any]],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """평가셋에 대해 Classic IR 지표를 계산한다."""
    if k_values is None:
        k_values = _K_VALUES

    max_k = max(k_values)
    metric_fns = {
        "hit_rate": hit_rate,
        "mrr": mrr,
        "precision": precision_at_k,
        "ndcg": ndcg_at_k,
    }

    # 카테고리별 결과 수집
    scores: dict[str, list[dict[str, float]]] = {}
    for cat in {"hr", "ops", "benefit"}:
        scores[cat] = []

    for item in dataset:
        results = await searcher.search(
            query=item["question"],
            top_k=max_k,
            category=item["category"],
        )
        retrieved_ids_raw = [r["doc_id"] for r in results]
        seen: set[str] = set()
        retrieved_ids: list[str] = []
        for did in retrieved_ids_raw:
            if did not in seen:
                seen.add(did)
                retrieved_ids.append(did)
        gt_ids = item["ground_truth_doc_ids"]

        item_scores: dict[str, float] = {}
        for k in k_values:
            for name, fn in metric_fns.items():
                item_scores[f"{name}@{k}"] = fn(
                    gt_ids, retrieved_ids, k
                )

        category = item["category"]
        if category in scores:
            scores[category].append(item_scores)

    # 카테고리별 평균
    by_category: dict[str, dict[str, float]] = {}
    all_scores: list[dict[str, float]] = []

    for cat, cat_scores in scores.items():
        if not cat_scores:
            continue
        keys = cat_scores[0].keys()
        by_category[cat] = {
            key: sum(s[key] for s in cat_scores)
            / len(cat_scores)
            for key in keys
        }
        all_scores.extend(cat_scores)

    # 전체 평균
    overall: dict[str, float] = {}
    if all_scores:
        keys = all_scores[0].keys()
        overall = {
            key: sum(s[key] for s in all_scores)
            / len(all_scores)
            for key in keys
        }

    return {
        "overall": overall,
        "by_category": by_category,
    }
