import logging
import re

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


def _build_summary_prompt(query: str, rows: list) -> str:
    return (
        f"질문: {query}\n\n"
        f"SQL 결과:\n{rows}\n\n"
        "결과를 자연어로 간단히 요약해줘."
    )


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
        """사내 MySQL DB를 조회할 때 사용합니다.
        회원 수, 거래 현황, 수익 지급 이력 등 내부 데이터의
        정량 질문에 사용하세요.
        """
        logger.info("[text_to_sql_tool] query=%r", query)

        schemas = await schema_searcher.search(query)
        if not schemas:
            logger.info(
                "[text_to_sql_tool] 관련 스키마 없음"
            )
            return _CANNOT_ANSWER
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
            logger.info(
                "[text_to_sql_tool] SQL 검증 실패"
            )
            return _CANNOT_ANSWER

        rows = await mysql_client.execute_select(sql)
        if rows is None:
            return _CANNOT_ANSWER
        if len(rows) == 0:
            return _NO_DATA

        summary_response = await llm.ainvoke(
            _build_summary_prompt(query, rows)
        )
        return _extract_llm_text(summary_response).strip()

    return text_to_sql_tool
