"""KasaNova RAG 평가 프레임워크 — CLI 진입점."""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s %(name)s | %(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_DATASET_PATH = (
    Path(__file__).parent / "dataset" / "eval_dataset.json"
)
_RESULTS_DIR = Path(__file__).parent / "results"


def _print_ir_report(
    ir_results: dict[str, Any],
) -> None:
    """Classic IR 결과를 콘솔에 출력한다."""
    print("\n" + "=" * 60)
    print("  Classic IR 평가 결과")
    print("=" * 60)

    for section, metrics in ir_results.items():
        if section == "overall":
            print(f"\n  [전체]")
        else:
            print(f"\n  [{section}]")
            metrics = ir_results["by_category"].get(
                section, {}
            )
            if not metrics:
                continue

        if section == "by_category":
            for cat, cat_metrics in metrics.items():
                print(f"\n  [{cat}]")
                for k, v in sorted(cat_metrics.items()):
                    print(f"    {k:20s} {v:.4f}")
            continue

        for k, v in sorted(metrics.items()):
            print(f"    {k:20s} {v:.4f}")


def _print_ragas_report(
    ragas_results: dict[str, Any],
) -> None:
    """RAGAS 결과를 콘솔에 출력한다."""
    print("\n" + "=" * 60)
    print("  RAGAS 평가 결과")
    print("=" * 60)

    overall = ragas_results.get("overall", {})
    print("\n  [전체]")
    for k, v in sorted(overall.items()):
        print(f"    {k:20s} {v:.4f}")

    by_cat = ragas_results.get("by_category", {})
    for cat, metrics in by_cat.items():
        print(f"\n  [{cat}]")
        for k, v in sorted(metrics.items()):
            print(f"    {k:20s} {v:.4f}")


def _print_miss_summary(
    score_stats: dict[str, Any],
) -> None:
    """Hit@1 실패 요약을 콘솔에 출력한다."""
    overall = score_stats.get("overall", {})
    miss_count = overall.get("miss_count", 0)
    miss_ids = overall.get("miss_query_ids", [])

    print("\n" + "-" * 60)
    print(f"  Hit@1 실패: {miss_count}건")
    if miss_ids:
        print(f"  실패 쿼리: {', '.join(miss_ids)}")
    print("-" * 60)


def _create_clients(
    provider: str, api_key: str
) -> tuple[Any, Any, Any]:
    """provider에 따라 qa_client, llm, judge_llm을 생성한다."""
    if provider == "claude":
        import anthropic
        from langchain_anthropic import ChatAnthropic

        qa_client = anthropic.Anthropic(api_key=api_key)
        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
        )
        judge_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
        )
    else:
        import google.generativeai as genai
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
        )

        genai.configure(api_key=api_key)
        qa_client = genai.GenerativeModel(
            "gemini-2.0-flash"
        )
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
        )
        judge_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
        )

    return qa_client, llm, judge_llm


async def _run(args: argparse.Namespace) -> None:
    """평가 실행 메인 로직."""
    from opensearchpy import AsyncOpenSearch

    from app.core.config import settings
    from app.rag.embedder import Embedder
    from app.rag.searcher import HybridSearcher

    provider = args.provider
    qa_client, llm, judge_llm = _create_clients(
        provider, args.api_key
    )

    os_client = AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        use_ssl=False,
        verify_certs=False,
    )

    try:
        embedder = Embedder()
        searcher = HybridSearcher(
            os_client, embedder, settings.opensearch_index
        )

        # 1. 평가셋 생성/로드
        if not args.ragas_only:
            from eval.generate_dataset import (
                generate_dataset,
            )

            dataset = await generate_dataset(
                os_client,
                qa_client,
                _DATASET_PATH,
                regenerate=args.regenerate,
                provider=provider,
            )
        else:
            if not _DATASET_PATH.exists():
                logger.error(
                    "평가셋이 없습니다: %s", _DATASET_PATH
                )
                sys.exit(1)
            dataset = json.loads(
                _DATASET_PATH.read_text()
            )

        logger.info(
            "평가셋 로드 완료: %d건", len(dataset)
        )

        results: dict[str, Any] = {
            "run_at": datetime.now().isoformat(),
            "dataset_size": len(dataset),
            "provider": provider,
        }

        # 2. Classic IR 평가
        if not args.ragas_only:
            from eval.evaluate_retrieval import (
                evaluate_retrieval,
            )

            logger.info("Classic IR 평가 시작")
            ir_results = await evaluate_retrieval(
                searcher, dataset
            )
            results["classic_ir"] = {
                "overall": ir_results["overall"],
                "by_category": ir_results["by_category"],
            }
            results["query_results"] = ir_results[
                "query_results"
            ]
            results["score_stats"] = ir_results[
                "score_stats"
            ]
            _print_ir_report(results["classic_ir"])
            _print_miss_summary(
                ir_results["score_stats"]
            )

        # 3. RAGAS 평가
        if not args.retrieval_only:
            from eval.evaluate_pipeline import (
                KureEmbeddings,
                evaluate_pipeline,
            )

            logger.info("RAGAS 평가 시작")
            kure_embeddings = KureEmbeddings()

            ragas_results = await evaluate_pipeline(
                searcher,
                llm,
                judge_llm,
                dataset,
                embeddings=kure_embeddings,
            )
            results["ragas"] = ragas_results
            _print_ragas_report(ragas_results)

        # 4. 결과 저장
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        result_path = (
            _RESULTS_DIR / f"{timestamp}_results.json"
        )
        result_path.write_text(
            json.dumps(
                results, ensure_ascii=False, indent=2
            )
        )
        logger.info("결과 저장: %s", result_path)

        print("\n" + "=" * 60)
        print(f"  결과 파일: {result_path}")
        print("=" * 60)

    finally:
        await os_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KasaNova RAG 평가 프레임워크"
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Classic IR 평가만 실행",
    )
    parser.add_argument(
        "--ragas-only",
        action="store_true",
        help="RAGAS 평가만 실행",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="평가셋 강제 재생성",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["claude", "gemini"],
        default="claude",
        help="LLM 프로바이더 선택 (기본: claude)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="LLM API Key (Claude 또는 Gemini)",
    )
    args = parser.parse_args()

    if args.retrieval_only and args.ragas_only:
        parser.error(
            "--retrieval-only와 --ragas-only는"
            " 동시에 사용할 수 없습니다."
        )

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
