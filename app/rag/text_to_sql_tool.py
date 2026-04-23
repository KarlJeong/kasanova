"""Text-to-SQL LangGraph tool (분해 + 실행 wrapper).

단일 조건 쿼리의 실제 파이프라인은 `text_to_sql_graph.py`의
컴파일된 StateGraph가 담당한다. 이 모듈은 그 위에 두 가지를 얹는다:

1. `query_planner.plan_query()`로 분해 필요 여부 판단
2. 분해 경로에서 모든 sub-query(filter + data)를 1라운드 병렬 실행하고,
   filter 서브쿼리의 key_column 기준 set 연산으로 최종 키를 결정한 뒤
   모든 서브쿼리의 rows를 key 기준으로 메모리에서 merge

분해가 필요 없으면 subgraph 1회 호출로 기존 동작과 동일하게 작동.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from app.db.mysql_client import MySQLClient
from app.rag.query_planner import QueryPlan, combine_keys, plan_query
from app.rag.schema_searcher import SchemaSearcher
from app.rag.searcher import HybridSearcher
from app.rag.table_companions import load_companion_tables
from app.rag.text_to_sql_graph import (
    TextToSqlDeps,
    TextToSqlState,
    build_text_to_sql_graph,
)

_COMPANIONS_PATH = (
    Path(__file__).resolve().parent / "table_companions.yaml"
)
from app.rag.text_to_sql_helpers import (
    CANNOT_ANSWER,
    NO_DATA,
    format_rows_for_tool_result,
)

# ── backward-compat re-exports (테스트 / 외부 import 용) ──────────────
from app.rag.text_to_sql_helpers import (  # noqa: F401
    extract_kb_tables as _extract_kb_tables,
)
from app.rag.text_to_sql_helpers import (  # noqa: F401
    format_rows_for_tool_result as _format_rows_for_tool_result,
)

logger = logging.getLogger(__name__)


async def _run_subgraph_for_rows(
    graph: CompiledStateGraph,
    query: str,
    literals: list[str],
    query_label: str = "",
) -> tuple[
    list[dict[str, Any]] | None, str | None, str | None
]:
    """subgraph를 한 번 호출하고 최종 state에서 rows를 뽑아 반환.

    Returns:
        (rows, key_column, error_message)
        - 성공: (rows, key_column, None)
        - 실패: (None, None, error_message)
    """
    initial_state: TextToSqlState = {
        "query": query,
        "literals": literals,
        "query_label": query_label,
    }
    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception:
        logger.exception(
            "[_run_subgraph_for_rows] subgraph 실행 예외:"
            " query=%r literals=%r",
            query,
            literals,
        )
        return None, None, CANNOT_ANSWER

    # graph 내부가 error를 state["error"]로 표시한 경우
    err = final_state.get("error")
    if err:
        return None, None, err

    rows = final_state.get("rows")
    if rows is None:
        return None, None, CANNOT_ANSWER
    key_column = final_state.get("key_column")
    return rows, key_column, None


async def _execute_decomposed(
    graph: CompiledStateGraph,
    plan: QueryPlan,
    literals: list[str],
) -> str:
    """plan에 따라 모든 sub-query를 1라운드 병렬 실행 후 결합한다.

    filter role 서브쿼리 → key_set 추출 → set 연산 → final_keys
    모든 서브쿼리(filter + data) → key 기준 rows merge
    """
    assert plan.requires_decomposition
    assert plan.combine is not None

    filter_subs = [
        s for s in plan.subqueries if s.role == "filter"
    ]
    data_subs = [
        s for s in plan.subqueries if s.role == "data"
    ]
    logger.info(
        "[_execute_decomposed] 분해 실행 시작:"
        " filter=%d, data=%d, combine=%s",
        len(filter_subs),
        len(data_subs),
        plan.combine,
    )

    # Soft validation: 공유 제약(= literals)이 일부 sub-question에서
    # 누락되면 planner가 공통 컨텍스트를 전파하지 못한 것이다.
    for lit in literals:
        missing = [
            sub for sub in plan.subqueries
            if lit not in sub.query
        ]
        if missing:
            logger.warning(
                "[_execute_decomposed] literal %r이(가) 일부"
                " sub-question 본문에 누락됨 (%d개) — planner가"
                " 공유 제약을 전파하지 못했을 가능성. 누락된"
                " sub-question: %s",
                lit,
                len(missing),
                [s.query for s in missing],
            )

    # ── 1라운드: 모든 서브쿼리 병렬 실행 ──────────────────────
    total = len(plan.subqueries)
    sub_results = await asyncio.gather(
        *[
            _run_subgraph_for_rows(
                graph, sub.query, literals,
                query_label=f"Q{i}/{total}",
            )
            for i, sub in enumerate(plan.subqueries, 1)
        ],
        return_exceptions=True,
    )

    # ── 결과 수집 ────────────────────────────────────────────
    filter_key_sets: list[set[Any]] = []
    sub_row_lists: list[list[dict[str, Any]]] = []
    resolved_keys: list[str] = []
    primary_key: str | None = None

    for i, result in enumerate(sub_results, 1):
        sub = plan.subqueries[i - 1]
        if isinstance(result, BaseException):
            logger.warning(
                "[_execute_decomposed] [%d] 예외: %s"
                " | role=%s sub=%r",
                i, result, sub.role, sub.query,
            )
            if sub.role == "filter":
                return CANNOT_ANSWER
            sub_row_lists.append([])
            resolved_keys.append("")
            continue

        rows, key_col, err = result
        if rows is None:
            logger.warning(
                "[_execute_decomposed] [%d] 실패: %s"
                " | role=%s sub=%r",
                i, err, sub.role, sub.query,
            )
            if sub.role == "filter":
                return CANNOT_ANSWER
            sub_row_lists.append([])
            resolved_keys.append("")
            continue

        if not rows:
            logger.info(
                "[_execute_decomposed] [%d] 결과 0건"
                " | role=%s sub=%r",
                i, sub.role, sub.query,
            )
            if sub.role == "filter":
                filter_key_sets.append(set())
            sub_row_lists.append([])
            resolved_keys.append(key_col or "")
            continue

        columns = list(rows[0].keys())
        if not key_col or key_col not in columns:
            logger.warning(
                "[_execute_decomposed] [%d] key 컬럼 %r이(가)"
                " 결과에 없음 (있는 컬럼: %s)"
                " | role=%s sub=%r",
                i, key_col, columns, sub.role, sub.query,
            )
            if sub.role == "filter":
                return CANNOT_ANSWER
            sub_row_lists.append([])
            resolved_keys.append("")
            continue

        resolved_keys.append(key_col)
        sub_row_lists.append(rows)

        if sub.role == "filter":
            keys = {
                row[key_col]
                for row in rows
                if row.get(key_col) is not None
            }
            filter_key_sets.append(keys)
            if primary_key is None:
                primary_key = key_col

        logger.info(
            "[_execute_decomposed] [%d/%d] 완료:"
            " %d행 (key_column=%s)"
            " | role=%s sub=%r",
            i, total, len(rows), key_col,
            sub.role, sub.query,
        )

    # ── set 연산 (filter만) ──────────────────────────────────
    final_keys = combine_keys(filter_key_sets, plan.combine)
    logger.info(
        "[_execute_decomposed] 결합(%s) 결과: %d개 키",
        plan.combine,
        len(final_keys),
    )

    if not final_keys:
        return NO_DATA
    if not primary_key:
        return CANNOT_ANSWER

    # ── 모든 서브쿼리 rows를 key 기준으로 merge ──────────────
    merged: dict[Any, dict[str, Any]] = {}
    for i, sub in enumerate(plan.subqueries):
        rows = sub_row_lists[i]
        key_col = resolved_keys[i]
        if not rows or not key_col:
            continue
        for row in rows:
            k = row.get(key_col)
            if k is None or k not in final_keys:
                continue
            if k not in merged:
                merged[k] = {primary_key: k}
            for col, val in row.items():
                if col == key_col and col != primary_key:
                    continue
                if col not in merged[k]:
                    merged[k][col] = val

    filtered = list(merged.values())

    filter_summary = ", ".join(
        f"F{i + 1}={len(s)}"
        for i, s in enumerate(filter_key_sets)
    )
    data_summary = (
        f", data={len(data_subs)}" if data_subs else ""
    )
    result_str = format_rows_for_tool_result(filtered)
    decomposed_str = (
        f"[분해 실행: {plan.combine},"
        f" {filter_summary}{data_summary}"
        f" → 최종 {len(final_keys)}건] {result_str}"
    )
    logger.info(
        "[_execute_decomposed] tool 결과: %s",
        decomposed_str[:300],
    )
    return decomposed_str


def create_text_to_sql_tool(
    schema_searcher: SchemaSearcher,
    kb_searcher: HybridSearcher,
    mysql_client: MySQLClient,
    llm: BaseChatModel,
) -> BaseTool:
    """Text-to-SQL LangGraph tool을 생성한다.

    단일 조건 쿼리는 `text_to_sql_graph`의 컴파일된 StateGraph를
    1회 호출한다. 멀티 조건 쿼리는 `query_planner`로 분해 후 모든
    sub-query를 1라운드 병렬 실행하고, filter role의 결과로 set 연산,
    전체 결과를 key 기준 merge한다.
    """
    companion_tables = load_companion_tables(_COMPANIONS_PATH)
    deps = TextToSqlDeps(
        schema_searcher=schema_searcher,
        kb_searcher=kb_searcher,
        mysql_client=mysql_client,
        llm=llm,
        companion_tables=companion_tables,
    )
    graph = build_text_to_sql_graph(deps)

    @tool
    async def text_to_sql_tool(
        query: str,
        literals: list[str] | None = None,
    ) -> str:
        """사내 MySQL DB를 조회해 정량 질문에 답할 때 사용하는 도구.

        회원 수, 거래 건수, 수익 지급 금액 등 숫자·집계·필터 기반의 내부 DB 질문에 적합하다.
        문서 지식·정책·가이드 질문은 retrieval_tool을 사용하라.

        query (필수):
        - 사용자 질문의 **의도**만 자연어로 담아라.
        - 다음 식별용 값은 query에 넣지 말고 literals로 분리하라 (RAG 검색 품질을 떨어뜨리기 때문):
            · DABS 종목코드 (예: "KR011A200005X8")
            · DABS/종목/건물 고유명 (예: "북촌 월하재", "역삼 한국빌딩")
            · 회원 ID·이메일·사람 이름
            · 주소
            · 구체적 숫자·금액·날짜
        - 한국어 질문은 한국어 그대로. SQL·영어 의역 금지.
        - 이 도구 내부에서 스키마 검색 후 LLM이 SQL을 생성한다. 호출자(상위 LLM)가 SQL을 만들면 안 된다.

        literals (선택):
        - SQL의 WHERE 절 등에 그대로 들어갈 식별용 값들의 리스트.
        - 예: DABS 종목코드, 종목 고유명, 회원 이메일, 특정 날짜 등.
        - 없으면 생략하거나 빈 리스트를 넘겨라.
        - **DABS 종목코드(예: `KR...`)를 아는 경우에는 종목명·건물명은 literals에 넣지 말고 코드만 넘겨라.**
          코드와 이름을 동시에 넘기면 SQL이 이름 컬럼으로 필터링할 위험이 있다.
          코드를 모르는 경우에 한해 이름을 넘겨도 된다.

        올바른 사용 예시:
        - query="이번 달 신규 가입 회원 수는?"
        - query="이 종목을 청약한 멤버 목록을 조회해줘", literals=["KR011A200005X8"]
        - query="지난주 DABS별 거래 금액 합계"

        잘못된 사용 예시 (금지):
        - query="SELECT COUNT(*) FROM kasa_member WHERE ..."
        - query="KR011A200005X8를 청약한 멤버 목록"  # 코드를 query에 포함
        - query="북촌 월하재를 청약한 멤버 목록"     # 고유명을 query에 포함
        """
        literals = literals or []
        logger.info(
            "[text_to_sql_tool] query=%r, literals=%r",
            query,
            literals,
        )

        # 1단계: 쿼리 분해 가능성 판단 (planner LLM 호출)
        plan = await plan_query(query, literals, llm)

        # 2단계 (분해 경로): 모든 sub-query를 1라운드 병렬 실행 후 결합
        if plan.requires_decomposition:
            logger.info(
                "[text_to_sql_tool] 분해 실행으로 분기 (reason=%s)",
                plan.reasoning,
            )
            return await _execute_decomposed(graph, plan, literals)

        # 2단계 (단일 경로): 기존 subgraph 1회 호출
        logger.info(
            "[text_to_sql_tool] 단일 SQL 경로 (reason=%s)",
            plan.reasoning,
        )
        initial_state: TextToSqlState = {
            "query": query,
            "literals": literals,
        }
        final_state = await graph.ainvoke(initial_state)
        return final_state.get("final_result", CANNOT_ANSWER)

    return text_to_sql_tool
