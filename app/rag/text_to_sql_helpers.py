"""text_to_sql 파이프라인에서 그래프 노드와 tool wrapper가 공유하는
순수 헬퍼·상수 모듈.

순수 함수만 모았기 때문에 의존성 그래프에서 leaf 위치에 있고, graph
모듈과 tool wrapper 어느 쪽도 이 파일을 안전하게 import 할 수 있다.
"""
from __future__ import annotations

import json
import re
from typing import Any

# ────────────────────────────────────────────────────────────────────
# 정규식 / 상수
# ────────────────────────────────────────────────────────────────────

FORBIDDEN_PATTERN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)\b",
    re.IGNORECASE,
)
CODE_FENCE_PATTERN = re.compile(
    r"^```[a-z]*\s*\n?|\n?```\s*$",
    re.IGNORECASE | re.MULTILINE,
)
KB_TABLE_PATTERN = re.compile(r"\bkasa_[a-z0-9_]+")

CANNOT_ANSWER = "조회할 수 없습니다"
NO_DATA = "조회된 데이터가 없습니다"
MAX_PREVIEW_ROWS = 1000
MAX_KB_PINNED_TABLES = 8

SQL_SYSTEM_PROMPT = """\
너는 사내 MySQL 8.x 전용 Text-to-SQL 생성기다.
사용자의 자연어 질문과 함께 주어진 테이블 스키마만을 근거로 SQL을 만든다.

[출력 계약 — 반드시 준수]
- 성공 시: JSON 한 객체만 출력한다 (코드 펜스·설명·주석·마크다운 금지).
  형식: {"sql": "SELECT ...", "key_column": "컬럼명"}
  - sql: SELECT로 시작하는 SQL 한 문장.
  - key_column: SELECT 절에서 결과 행을 고유하게 식별하는 대표 키 컬럼명.
    별칭(AS)을 사용했다면 별칭을 적는다.
    집계 쿼리(COUNT, SUM 등)처럼 식별 키가 없으면 null.
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
- 식별자 우선순위: 동일 대상에 대해 **코드**(예: DABS 종목코드 `KR...`)와 **이름**(종목명·건물명·프로젝트명 등)이 모두 리터럴로 주어지면 **반드시 코드 컬럼으로만 필터링**한다. 이 경우 이름 컬럼은 `=`·`LIKE`·`IN` 어디에도 사용하지 않는다. 코드가 없고 이름만 주어진 경우에 한해 이름 컬럼으로 필터링한다.
- 집계 질문은 SELECT 절의 non-aggregate 컬럼을 모두 `GROUP BY`에 포함한다.
- 가독성을 위해 여러 테이블을 다룰 때 테이블 별칭을 사용한다.
- MySQL 8.x 방언만 사용한다. 타 DB 방언 함수(`DATE_TRUNC`, `TO_DATE` 등) 금지.

[판단 규칙]
- 필요한 테이블·컬럼이 주어진 스키마에 없으면 추측하지 말고 `UNKNOWN: <사유>`로 출력한다.
- 질문이 모호해도 가장 합리적인 해석 하나로 SQL을 생성한다. 되묻거나 복수 해석을 나열하지 않는다.
"""


# ────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ────────────────────────────────────────────────────────────────────


def strip_code_fence(text: str) -> str:
    return CODE_FENCE_PATTERN.sub("", text).strip()


def parse_unknown_reason(text: str) -> str | None:
    """LLM 응답이 UNKNOWN 계열이면 사유를 추출한다."""
    stripped = text.strip()
    if not stripped:
        return "빈 응답"
    upper = stripped.upper()
    if upper == "UNKNOWN":
        return "사유 미기재"
    if upper.startswith("UNKNOWN"):
        reason = stripped[len("UNKNOWN") :].lstrip(" :-\n\t")
        return reason or "사유 미기재"
    return None


def validate_sql(sql: str) -> bool:
    """SELECT로 시작하고 금지 키워드가 없는지 검증."""
    if not sql:
        return False
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        return False
    if FORBIDDEN_PATTERN.search(stripped):
        return False
    return True


def format_schemas(schemas: list[dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(s["schema"] for s in schemas)


def extract_kb_tables(content: str) -> list[str]:
    """KB 문서 본문에서 `kasa_*` 테이블명을 등장 순서대로 dedup."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in KB_TABLE_PATTERN.finditer(content):
        name = match.group(0)
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def build_sql_prompt(
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
        "JSON을 출력하라. "
        '형식: {"sql": "SELECT ...", "key_column": "컬럼명"} '
        "key_column은 SELECT 절에서 행을 식별하는 대표 키 컬럼명이다. "
        "집계 쿼리라 식별 키가 없으면 null. "
        "질문에 명시된 식별자·코드·이름·날짜 등은 위 리터럴 목록에 "
        "있는 값을 그대로 사용한다. "
        "리터럴에 DABS 종목코드(예: `KR...`)와 종목명·건물명이 함께 "
        "있으면 **코드 컬럼으로만** 필터링하고 이름 컬럼은 사용하지 "
        "말라. 코드가 없을 때에 한해 이름으로 필터링한다. "
        "도메인 참고 블록은 테이블·JOIN 선택이 모호할 때만 참고용으로 "
        "활용하고, 질문과 관련 없어 보이면 완전히 무시하라. "
        "스키마만으로 답할 수 없으면 정확히 `UNKNOWN`만 출력하라."
    )


def format_rows_for_tool_result(
    rows: list[dict[str, Any]],
) -> str:
    """MySQL 결과를 상위 LLM이 그대로 활용할 수 있는 형태로 직렬화.

    SQL을 절대 포함하지 않는 정해진 포맷으로 만들어, inner-LLM 요약
    단계에서 SQL이 누출되는 경로를 원천 차단한다.
    """
    preview = rows[:MAX_PREVIEW_ROWS]
    truncated = len(rows) > MAX_PREVIEW_ROWS

    if len(rows) == 1 and len(rows[0]) == 1:
        only_value = next(iter(rows[0].values()))
        return f"결과 1건: {only_value}"

    rows_json = json.dumps(
        preview, ensure_ascii=False, default=str
    )
    suffix = " (상위 1000건만 표시)" if truncated else ""
    return f"결과 {len(rows)}건{suffix}: {rows_json}"


def parse_sql_response(
    text: str,
) -> tuple[str | None, str | None]:
    """LLM 응답에서 SQL과 key_column을 추출한다.

    JSON 형식이면 sql/key_column을 파싱하고,
    plain SQL이면 (sql, None)으로 폴백한다.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            return data.get("sql"), data.get("key_column")
        except json.JSONDecodeError:
            pass
    return stripped if stripped else None, None


def extract_llm_text(response: Any) -> str:
    content = getattr(response, "content", response)
    return str(content)
