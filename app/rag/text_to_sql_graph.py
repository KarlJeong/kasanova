"""Text-to-SQL 단계별 LangGraph subgraph.

기존 `text_to_sql_tool`의 모놀리식 함수를 명시적인 노드 그래프로
재배치한다. 이 단계에서는 동작이 변하지 않는다 — 단일 조건 쿼리에
대한 schema 검색 → KB 보강 → SQL 생성 → 검증 → 실행 → 결과 포맷
파이프라인을 그대로 옮기되, 각 단계가 독립 노드가 된다.

분해 경로(Pattern B)는 이 subgraph 안이 아니라 `text_to_sql_tool.py`
wrapper 레이어에서 처리한다. wrapper가 `query_planner.plan_query()`로
분해 여부를 결정한 뒤, sub-query마다 이 subgraph를 독립적으로 태우고
결과 rows를 코드가 set 연산으로 결합한다. 이렇게 두면 subgraph는
"단일 조건 쿼리 1회 실행" 의미를 유지할 수 있어 재사용·테스트가
깔끔해진다.

상태(state) 필드는 노드 사이의 유일한 통신 채널이다. 닫힘(closure)
대신 명시적 상태를 쓰는 것이 디버깅·테스트·향후 확장(분해 경로의
sub-query가 KB/schema를 공유)에 유리하다.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.db.mysql_client import MySQLClient
from app.rag.schema_searcher import SchemaSearcher
from app.rag.searcher import HybridSearcher
from app.rag.text_to_sql_helpers import (
    CANNOT_ANSWER,
    MAX_KB_PINNED_TABLES,
    NO_DATA,
    SQL_SYSTEM_PROMPT,
    build_sql_prompt,
    extract_kb_tables,
    extract_llm_text,
    format_rows_for_tool_result,
    format_schemas,
    parse_sql_response,
    parse_unknown_reason,
    strip_code_fence,
    validate_sql,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# State / Deps
# ────────────────────────────────────────────────────────────────────


class TextToSqlState(TypedDict, total=False):
    """단계별 그래프 상태.

    `total=False` — 각 노드는 자기가 채우는 필드만 부분 갱신해서
    반환한다. 초기 상태는 query/literals만 들어 있다.
    """

    # 입력 (그래프 시작 시 채워짐)
    query: str
    literals: list[str]
    query_label: str  # 로그 접두사 (예: "Q1/3") — 분해 시 구분용

    # retrieve_context_node 출력
    schemas: list[dict[str, Any]]
    kb_docs: list[dict[str, Any]]

    # generate_sql_node 출력
    sql: str | None
    key_column: str | None  # SQL SELECT의 대표 키 컬럼명

    # execute_sql_node 출력
    rows: list[dict[str, Any]] | None

    # 최종 결과 (format_result_node가 채움)
    final_result: str
    error: str | None  # 단락 회로용 — 채워지면 곧장 format_result로


@dataclass(frozen=True)
class TextToSqlDeps:
    """그래프 노드들이 공유하는 외부 의존성.

    state가 아니라 closure로 바인딩한다. 매 호출 사이에 변하지 않고,
    state schema에 노출할 이유가 없는 값들이다.
    """

    schema_searcher: SchemaSearcher
    kb_searcher: HybridSearcher
    mysql_client: MySQLClient
    llm: BaseChatModel
    companion_tables: dict[str, list[str]] = field(
        default_factory=dict
    )


# ────────────────────────────────────────────────────────────────────
# Routing
# ────────────────────────────────────────────────────────────────────


def _route_after_retrieve(state: TextToSqlState) -> str:
    if state.get("error"):
        return "format_result"
    return "augment_schemas"


def _route_after_generate(state: TextToSqlState) -> str:
    if state.get("error"):
        return "format_result"
    return "execute_sql"


# ────────────────────────────────────────────────────────────────────
# Node factories
# ────────────────────────────────────────────────────────────────────


def _make_retrieve_context_node(deps: TextToSqlDeps):
    async def retrieve_context_node(
        state: TextToSqlState,
    ) -> dict[str, Any]:
        query = state["query"]
        lbl = state.get("query_label", "")
        pfx = f"[{lbl}][retrieve_context]" if lbl else "[retrieve_context]"
        logger.info(
            "%s query=%r literals=%r",
            pfx,
            query,
            state.get("literals", []),
        )

        # 스키마와 KB를 병렬 조회 (둘 다 같은 query 기반).
        schemas_result, kb_result = await asyncio.gather(
            deps.schema_searcher.search(query, log_prefix=lbl),
            deps.kb_searcher.fetch_best_docs(
                query, category="kb"
            ),
            return_exceptions=True,
        )

        if isinstance(schemas_result, BaseException):
            logger.exception(
                "%s 스키마 조회 실패", pfx,
                exc_info=schemas_result,
            )
            return {"error": CANNOT_ANSWER}
        schemas: list[dict[str, Any]] = schemas_result

        if isinstance(kb_result, BaseException):
            logger.exception(
                "%s KB 조회 실패", pfx,
                exc_info=kb_result,
            )
            kb_docs: list[dict[str, Any]] = []
        else:
            kb_docs = kb_result

        if not schemas:
            logger.info("%s 관련 스키마 없음", pfx)
            return {
                "schemas": [],
                "kb_docs": kb_docs,
                "error": CANNOT_ANSWER,
            }

        logger.info(
            "%s 스키마 %d개 후보 (RAG): %s",
            pfx,
            len(schemas),
            [s["table_name"] for s in schemas],
        )

        # KB 본문 → kasa_* 합집합 → 누락 테이블 보강
        if kb_docs:
            kb_union: list[str] = []
            kb_seen: set[str] = set()
            for idx, doc in enumerate(kb_docs, 1):
                doc_tables = extract_kb_tables(doc["content"])
                added = 0
                for t in doc_tables:
                    if t in kb_seen:
                        continue
                    kb_seen.add(t)
                    kb_union.append(t)
                    added += 1
                logger.info(
                    "%s KB 문서 [%d/%d] %s"
                    " (score=%.4f) → 테이블 %d개 언급,"
                    " 신규 %d개",
                    pfx,
                    idx,
                    len(kb_docs),
                    doc["source"],
                    doc["score"],
                    len(doc_tables),
                    added,
                )

            existing = {s["table_name"] for s in schemas}
            missing = [
                t for t in kb_union if t not in existing
            ][:MAX_KB_PINNED_TABLES]
            logger.info(
                "%s KB 합집합 %d개,"
                " 기존 후보 외 %d개 보강 시도: %s",
                pfx,
                len(kb_union),
                len(missing),
                missing,
            )
            if missing:
                extra = (
                    await deps.schema_searcher.fetch_schemas_by_names(
                        missing, log_prefix=lbl,
                    )
                )
                if extra:
                    schemas = schemas + extra
                    logger.info(
                        "%s KB 기반 보강 후"
                        " 스키마 %d개: %s",
                        pfx,
                        len(schemas),
                        [s["table_name"] for s in schemas],
                    )

        return {"schemas": schemas, "kb_docs": kb_docs}

    return retrieve_context_node


def _make_augment_schemas_node(deps: TextToSqlDeps):
    async def augment_schemas_node(
        state: TextToSqlState,
    ) -> dict[str, Any]:
        lbl = state.get("query_label", "")
        pfx = f"[{lbl}][augment_schemas]" if lbl else "[augment_schemas]"

        schemas = state.get("schemas") or []
        mapping = deps.companion_tables
        if not schemas or not mapping:
            return {}

        existing = {s["table_name"] for s in schemas}
        to_inject: list[str] = []
        triggers_log: list[tuple[str, list[str]]] = []
        for s in schemas:
            buddies = mapping.get(s["table_name"]) or []
            added_here: list[str] = []
            for buddy in buddies:
                if buddy in existing or buddy in to_inject:
                    continue
                to_inject.append(buddy)
                added_here.append(buddy)
            if added_here:
                triggers_log.append(
                    (s["table_name"], added_here)
                )

        if not to_inject:
            logger.info(
                "%s 동반 테이블 트리거 없음", pfx,
            )
            return {}

        for trigger, added in triggers_log:
            logger.info(
                "%s 트리거 %s → 주입 대상 %s",
                pfx, trigger, added,
            )

        extra = await deps.schema_searcher.fetch_schemas_by_names(
            to_inject, log_prefix=lbl,
        )
        if not extra:
            logger.info(
                "%s 주입 대상 조회 결과 없음", pfx,
            )
            return {}

        new_schemas = schemas + extra
        logger.info(
            "%s 주입 후 스키마 %d개: %s",
            pfx,
            len(new_schemas),
            [s["table_name"] for s in new_schemas],
        )
        return {"schemas": new_schemas}

    return augment_schemas_node


def _make_generate_sql_node(deps: TextToSqlDeps):
    async def generate_sql_node(
        state: TextToSqlState,
    ) -> dict[str, Any]:
        query = state["query"]
        literals = state.get("literals", [])
        schemas = state.get("schemas", [])
        kb_docs = state.get("kb_docs", [])
        lbl = state.get("query_label", "")
        pfx = f"[{lbl}][generate_sql]" if lbl else "[generate_sql]"

        # 본문 주입은 top-1만 (멀티 도메인이라도 하나만 주입해
        # 프롬프트 비대화/혼동을 막는다).
        if kb_docs:
            primary = kb_docs[0]
            domain_knowledge = primary["content"]
            logger.info(
                "%s 도메인 지식 본문 주입 (top-1):"
                " %s (청크 %d개, %d자, score=%.4f)"
                " [총 KB %d개 중]",
                pfx,
                primary["source"],
                primary["chunk_count"],
                len(domain_knowledge),
                primary["score"],
                len(kb_docs),
            )
        else:
            domain_knowledge = ""
            logger.info("%s 도메인 지식 없음", pfx)

        schemas_text = format_schemas(schemas)
        sql_user_prompt = build_sql_prompt(
            schemas_text,
            domain_knowledge,
            query,
            literals,
        )
        sql_response = await deps.llm.ainvoke(
            [
                ("system", SQL_SYSTEM_PROMPT),
                ("human", sql_user_prompt),
            ]
        )
        raw_response = extract_llm_text(sql_response)
        cleaned = strip_code_fence(raw_response)

        unknown_reason = parse_unknown_reason(cleaned)
        if unknown_reason is not None:
            logger.warning(
                "%s SQL 생성 실패 (UNKNOWN): %s"
                " | query=%r literals=%r 후보 스키마=%s",
                pfx,
                unknown_reason,
                query,
                literals,
                [s["table_name"] for s in schemas],
            )
            return {"sql": None, "error": CANNOT_ANSWER}

        sql, key_column = parse_sql_response(cleaned)

        if not validate_sql(sql or ""):
            logger.warning(
                "%s SQL 검증 실패: %s", pfx, sql
            )
            return {"sql": None, "error": CANNOT_ANSWER}

        logger.info(
            "%s 생성 SQL: %s (key_column=%s)",
            pfx,
            sql,
            key_column,
        )
        return {"sql": sql, "key_column": key_column}

    return generate_sql_node


def _make_execute_sql_node(deps: TextToSqlDeps):
    async def execute_sql_node(
        state: TextToSqlState,
    ) -> dict[str, Any]:
        sql = state.get("sql")
        if not sql:
            # 정상 경로에서는 도달하지 않음 (라우팅이 막아줌).
            return {"error": CANNOT_ANSWER}

        lbl = state.get("query_label", "")
        pfx = f"[{lbl}][execute_sql]" if lbl else "[execute_sql]"
        logger.info("%s MySQL 실행: %s", pfx, sql)
        rows = await deps.mysql_client.execute_select(sql)
        if rows is None:
            return {"rows": None, "error": CANNOT_ANSWER}
        return {"rows": rows}

    return execute_sql_node


def _make_format_result_node():
    async def format_result_node(
        state: TextToSqlState,
    ) -> dict[str, Any]:
        lbl = state.get("query_label", "")
        pfx = f"[{lbl}][format_result]" if lbl else "[format_result]"
        # 단락 회로 — error가 있으면 그대로 반환
        if state.get("error"):
            err = state["error"]
            logger.info("%s 에러 경로: %s", pfx, err)
            return {"final_result": err}

        rows = state.get("rows")
        if rows is None:
            return {"final_result": CANNOT_ANSWER}
        if len(rows) == 0:
            return {"final_result": NO_DATA}

        result_str = format_rows_for_tool_result(rows)
        logger.info(
            "%s tool 결과: %s", pfx, result_str[:300]
        )
        return {"final_result": result_str}

    return format_result_node


# ────────────────────────────────────────────────────────────────────
# Builder
# ────────────────────────────────────────────────────────────────────


def build_text_to_sql_graph(
    deps: TextToSqlDeps,
) -> CompiledStateGraph:
    """TextToSqlState 기반 LangGraph subgraph를 빌드해 컴파일한다.

    팩토리 단위로 한 번만 호출하고, 컴파일된 그래프는 재사용한다.
    """
    builder = StateGraph(TextToSqlState)
    builder.add_node(
        "retrieve_context", _make_retrieve_context_node(deps)
    )
    builder.add_node(
        "augment_schemas", _make_augment_schemas_node(deps)
    )
    builder.add_node(
        "generate_sql", _make_generate_sql_node(deps)
    )
    builder.add_node(
        "execute_sql", _make_execute_sql_node(deps)
    )
    builder.add_node(
        "format_result", _make_format_result_node()
    )

    builder.add_edge(START, "retrieve_context")
    builder.add_conditional_edges(
        "retrieve_context",
        _route_after_retrieve,
        {
            "augment_schemas": "augment_schemas",
            "format_result": "format_result",
        },
    )
    builder.add_edge("augment_schemas", "generate_sql")
    builder.add_conditional_edges(
        "generate_sql",
        _route_after_generate,
        {
            "execute_sql": "execute_sql",
            "format_result": "format_result",
        },
    )
    builder.add_edge("execute_sql", "format_result")
    builder.add_edge("format_result", END)

    return builder.compile()
