import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from app.db.mysql_client import MySQLClient
from app.rag.schema_searcher import SchemaSearcher

logger = logging.getLogger(__name__)

_FORBIDDEN_PATTERN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)\b",
    re.IGNORECASE,
)
_CODE_FENCE_PATTERN = re.compile(
    r"^```(?:sql)?\s*\n?|\n?```\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_CANNOT_ANSWER = "조회할 수 없습니다"
_NO_DATA = "조회된 데이터가 없습니다"
_MAX_PREVIEW_ROWS = 10


def _strip_code_fence(text: str) -> str:
    return _CODE_FENCE_PATTERN.sub("", text).strip()


def _validate_sql(sql: str) -> bool:
    """SELECT로 시작하고 금지 키워드가 없는지 검증."""
    if not sql or sql.strip().upper() == "UNKNOWN":
        return False
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        return False
    if _FORBIDDEN_PATTERN.search(stripped):
        return False
    return True


def _format_schemas(schemas: list[dict]) -> str:
    return "\n\n".join(
        f"{s['table_name']}:\n{s['schema']}" for s in schemas
    )


def _build_sql_prompt(
    schemas_text: str, domain_knowledge: str, query: str
) -> str:
    kb_block = (
        domain_knowledge.strip() if domain_knowledge else "(없음)"
    )
    return (
        "[스키마]\n"
        f"{schemas_text}\n\n"
        "[도메인 지식]\n"
        f"{kb_block}\n\n"
        "[질문]\n"
        f"{query}\n\n"
        "규칙:\n"
        "- SELECT만 허용\n"
        "- 주어진 스키마의 테이블명/컬럼명만 사용\n"
        "- FK 관계에 명시된 조건으로만 JOIN\n"
        "- 설명 없이 SQL만 출력\n"
        "- 스키마만으로 답할 수 없으면 UNKNOWN 출력"
    )


def _format_rows_for_tool_result(
    rows: list[dict[str, Any]],
) -> str:
    """MySQL 결과를 상위 LLM이 그대로 활용할 수 있는 형태로 직렬화.

    SQL을 절대 포함하지 않는 정해진 포맷으로 만들어,
    inner-LLM 요약 단계에서 SQL이 누출되는 경로를 원천 차단한다.
    """
    preview = rows[:_MAX_PREVIEW_ROWS]
    truncated = len(rows) > _MAX_PREVIEW_ROWS

    # 단일 스칼라 결과는 값 하나로 단순화
    if len(rows) == 1 and len(rows[0]) == 1:
        only_value = next(iter(rows[0].values()))
        return f"결과 1건: {only_value}"

    rows_json = json.dumps(
        preview, ensure_ascii=False, default=str
    )
    suffix = " (상위 10건만 표시)" if truncated else ""
    return f"결과 {len(rows)}건{suffix}: {rows_json}"


def _extract_llm_text(response) -> str:
    content = getattr(response, "content", response)
    return str(content)


def create_text_to_sql_tool(
    schema_searcher: SchemaSearcher,
    retrieval_tool: BaseTool,
    mysql_client: MySQLClient,
    llm: BaseChatModel,
) -> BaseTool:
    """Text-to-SQL LangGraph 도구를 생성한다."""

    @tool
    async def text_to_sql_tool(query: str) -> str:
        """사내 MySQL DB를 조회해 정량 질문에 답할 때 사용하는 도구.

        회원 수, 거래 건수, 수익 지급 금액 등 숫자·집계·필터 기반의
        내부 DB 질문에 적합하다. 문서 지식·정책·가이드 질문은
        retrieval_tool을 사용하라.

        query (필수):
        - 반드시 사용자가 말한 **원본 자연어 질문을 그대로** 넘겨라.
        - 한국어 질문은 한국어 그대로. SQL·영어 의역·재작성 금지.
        - 이 도구 내부에서 스키마 검색 후 LLM이 SQL을 생성한다.
          호출자(상위 LLM)가 SQL을 만들면 안 된다.

        올바른 사용 예시:
        - query="이번 달 신규 가입 회원 수는?"
        - query="지난주 DABS별 거래 금액 합계"
        - query="현재 전체 활성 회원 수?"

        잘못된 사용 예시 (금지):
        - query="SELECT COUNT(*) FROM kasa_member WHERE ..."
        - query="How many active members are there?"
        """
        logger.info("[text_to_sql_tool] query=%r", query)

        schemas = await schema_searcher.search(query)
        if not schemas:
            logger.info(
                "[text_to_sql_tool] 관련 스키마 없음"
            )
            return _CANNOT_ANSWER
        logger.info(
            "[text_to_sql_tool] 스키마 %d개 후보: %s",
            len(schemas),
            [s["table_name"] for s in schemas],
        )
        schemas_text = _format_schemas(schemas)

        try:
            domain_knowledge = await retrieval_tool.ainvoke(
                {"query": query, "category": "kb"}
            )
        except Exception:
            logger.exception(
                "[text_to_sql_tool] 도메인 지식 조회 실패"
            )
            domain_knowledge = ""

        sql_prompt = _build_sql_prompt(
            schemas_text,
            str(domain_knowledge or ""),
            query,
        )
        sql_response = await llm.ainvoke(sql_prompt)
        sql = _strip_code_fence(
            _extract_llm_text(sql_response)
        )
        logger.info(
            "[text_to_sql_tool] 생성 SQL: %s", sql
        )

        if not _validate_sql(sql):
            logger.warning(
                "[text_to_sql_tool] SQL 검증 실패: %s", sql
            )
            return _CANNOT_ANSWER

        logger.info(
            "[text_to_sql_tool] MySQL 실행: %s", sql
        )
        rows = await mysql_client.execute_select(sql)
        if rows is None:
            return _CANNOT_ANSWER
        if len(rows) == 0:
            return _NO_DATA

        tool_result = _format_rows_for_tool_result(rows)
        logger.info(
            "[text_to_sql_tool] tool 결과: %s", tool_result
        )
        return tool_result

    return text_to_sql_tool
