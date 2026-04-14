import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from app.db.mysql_client import MySQLClient
from app.rag.schema_searcher import SchemaSearcher
from app.rag.searcher import HybridSearcher

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

_SQL_SYSTEM_PROMPT = """\
너는 사내 MySQL 8.x 전용 Text-to-SQL 생성기다.
사용자의 자연어 질문과 함께 주어진 테이블 스키마만을 근거로 SQL을 만든다.

[출력 계약 — 반드시 준수]
- 성공 시: 코드 펜스·설명·주석·마크다운 없이 `SELECT`로 시작하는 SQL 한 문장만 출력한다.
- 스키마만으로 답할 수 없으면 `UNKNOWN: <사유>` 형식으로 한 줄만 출력한다.
  사유는 한국어로 간결히 (예: `UNKNOWN: kasa_offering_subscription에 종목코드 컬럼이 없음`,
  `UNKNOWN: 멤버 이메일을 참조하는 테이블이 주어지지 않음`).
- 접두/접미 텍스트, 인사말, 가정 설명, 여러 문장 출력 금지.

[SQL 규칙]
- `SELECT`만 허용. `INSERT`·`UPDATE`·`DELETE`·`DROP`·`ALTER`·`TRUNCATE` 등 DDL/DML 전면 금지.
- 주어진 스키마에 존재하는 테이블명·컬럼명만 사용한다. 임의 추측·생성 금지.
- JOIN은 각 테이블 블록의 `관계:` 섹션에 명시된 FK 조건으로만 수행한다.
- `SELECT *` 금지. 컬럼을 명시적으로 나열한다.
- "목록"·"최근"·"상위" 류 질문은 `ORDER BY`와 적절한 `LIMIT`을 반드시 포함한다.
- datetime 비교는 KST 기준이며, `datetime(6)` 컬럼은 문자열 리터럴(예: '2026-04-14 00:00:00')로 비교한다.
- 문자열 매칭은 기본적으로 `=`을 사용하고, 부분 일치가 필요한 경우에만 `LIKE`를 사용한다.
- 집계 질문은 SELECT 절의 non-aggregate 컬럼을 모두 `GROUP BY`에 포함한다.
- 가독성을 위해 여러 테이블을 다룰 때 테이블 별칭을 사용한다.
- MySQL 8.x 방언만 사용한다. 타 DB 방언 함수(`DATE_TRUNC`, `TO_DATE` 등) 금지.

[판단 규칙]
- 필요한 테이블·컬럼이 주어진 스키마에 없으면 추측하지 말고 `UNKNOWN: <사유>`로 출력한다.
- 질문이 모호해도 가장 합리적인 해석 하나로 SQL을 생성한다. 되묻거나 복수 해석을 나열하지 않는다.
"""


def _strip_code_fence(text: str) -> str:
    return _CODE_FENCE_PATTERN.sub("", text).strip()


def _parse_unknown_reason(text: str) -> str | None:
    """LLM 응답이 UNKNOWN 계열이면 사유를 추출한다.

    반환값이 None이 아니면 SQL 생성 실패로 간주한다.
    사유가 명시되지 않았으면 빈 문자열이 아닌 기본 메시지를 반환한다.
    """
    stripped = text.strip()
    if not stripped:
        return "빈 응답"
    upper = stripped.upper()
    if upper == "UNKNOWN":
        return "사유 미기재"
    if upper.startswith("UNKNOWN"):
        # "UNKNOWN: ...", "UNKNOWN - ...", "UNKNOWN\n..." 등 허용
        reason = stripped[len("UNKNOWN"):].lstrip(" :-\n\t")
        return reason or "사유 미기재"
    return None


def _validate_sql(sql: str) -> bool:
    """SELECT로 시작하고 금지 키워드가 없는지 검증."""
    if not sql:
        return False
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        return False
    if _FORBIDDEN_PATTERN.search(stripped):
        return False
    return True


def _format_schemas(schemas: list[dict]) -> str:
    return "\n\n---\n\n".join(s["schema"] for s in schemas)


def _build_sql_prompt(
    schemas_text: str,
    domain_knowledge: str,
    query: str,
    literals: list[str],
) -> str:
    kb_block = (
        domain_knowledge.strip() if domain_knowledge else "(없음)"
    )
    if literals:
        literals_block = "\n".join(f"- {lit}" for lit in literals)
    else:
        literals_block = "(없음)"
    return (
        "## 사용 가능한 스키마\n"
        f"{schemas_text}\n\n"
        "## 도메인 참고 (질문과 무관하면 무시하라)\n"
        f"{kb_block}\n\n"
        "## 질문\n"
        f"{query}\n\n"
        "## 리터럴 (WHERE 절 등에 그대로 사용)\n"
        f"{literals_block}\n\n"
        "위 스키마에 정의된 테이블·컬럼만 사용해 질문에 답하는 "
        "MySQL SELECT 문 하나만 출력하라. "
        "질문에 명시된 식별자·코드·이름·날짜 등은 위 리터럴 목록에 "
        "있는 값을 그대로 사용한다. "
        "도메인 참고 블록은 테이블·JOIN 선택이 모호할 때만 참고용으로 "
        "활용하고, 질문과 관련 없어 보이면 완전히 무시하라. "
        "스키마만으로 답할 수 없으면 정확히 `UNKNOWN`만 출력하라."
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
    kb_searcher: HybridSearcher,
    mysql_client: MySQLClient,
    llm: BaseChatModel,
) -> BaseTool:
    """Text-to-SQL LangGraph 도구를 생성한다.

    kb_searcher: 도메인 지식(category="kb") 검색에 사용한다.
    청크 단위 하이브리드 검색으로 top-1 문서를 찾아 통째로 주입한다.
    """

    @tool
    async def text_to_sql_tool(
        query: str,
        literals: list[str] | None = None,
    ) -> str:
        """사내 MySQL DB를 조회해 정량 질문에 답할 때 사용하는 도구.

        회원 수, 거래 건수, 수익 지급 금액 등 숫자·집계·필터 기반의
        내부 DB 질문에 적합하다. 문서 지식·정책·가이드 질문은
        retrieval_tool을 사용하라.

        query (필수):
        - 사용자 질문의 **의도**만 자연어로 담아라.
        - DABS 종목코드·회원 ID·주소·사람 이름·구체적 숫자·날짜 등
          식별용 리터럴 값은 query에 넣지 말고 literals 파라미터로 분리하라.
          (스키마 RAG 검색 품질을 떨어뜨리기 때문)
        - 한국어 질문은 한국어 그대로. SQL·영어 의역 금지.
        - 이 도구 내부에서 스키마 검색 후 LLM이 SQL을 생성한다.
          호출자(상위 LLM)가 SQL을 만들면 안 된다.

        literals (선택):
        - SQL의 WHERE 절 등에 그대로 들어갈 식별용 값들의 리스트.
        - 예: DABS 종목코드("KR011A200005X8"), 회원 이메일, 특정 날짜 등.
        - 없으면 생략하거나 빈 리스트를 넘겨라.

        올바른 사용 예시:
        - query="이번 달 신규 가입 회원 수는?"
        - query="북촌 월하재를 청약한 멤버 목록을 조회해줘",
          literals=["KR011A200005X8"]
        - query="지난주 DABS별 거래 금액 합계"

        잘못된 사용 예시 (금지):
        - query="SELECT COUNT(*) FROM kasa_member WHERE ..."
        - query="KR011A200005X8를 청약한 멤버 목록"  # 코드를 query에 포함
        """
        literals = literals or []
        logger.info(
            "[text_to_sql_tool] query=%r, literals=%r",
            query,
            literals,
        )

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

        # 도메인 지식: top-1 문서만 통째로 주입.
        # 과거에는 retrieval_tool을 호출했으나 여러 kb 문서의 청크가
        # 뒤섞여 전달되는 문제가 있어, 문서 단위 승격으로 대체했다.
        try:
            kb_doc = await kb_searcher.fetch_best_doc(
                query, category="kb"
            )
        except Exception:
            logger.exception(
                "[text_to_sql_tool] 도메인 지식 조회 실패"
            )
            kb_doc = None

        if kb_doc:
            domain_knowledge = kb_doc["content"]
            logger.info(
                "[text_to_sql_tool] 도메인 지식 주입:"
                " %s (청크 %d개, %d자, score=%.4f)",
                kb_doc["source"],
                kb_doc["chunk_count"],
                len(domain_knowledge),
                kb_doc["score"],
            )
        else:
            domain_knowledge = ""
            logger.info(
                "[text_to_sql_tool] 도메인 지식 없음"
            )

        sql_user_prompt = _build_sql_prompt(
            schemas_text,
            str(domain_knowledge or ""),
            query,
            literals,
        )
        sql_response = await llm.ainvoke(
            [
                ("system", _SQL_SYSTEM_PROMPT),
                ("human", sql_user_prompt),
            ]
        )
        raw_response = _extract_llm_text(sql_response)
        logger.info(
            "[text_to_sql_tool] LLM 원본 응답: %s", raw_response
        )
        sql = _strip_code_fence(raw_response)

        unknown_reason = _parse_unknown_reason(sql)
        if unknown_reason is not None:
            logger.warning(
                "[text_to_sql_tool] SQL 생성 실패 (UNKNOWN): %s"
                " | query=%r literals=%r 후보 스키마=%s",
                unknown_reason,
                query,
                literals,
                [s["table_name"] for s in schemas],
            )
            return _CANNOT_ANSWER

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
