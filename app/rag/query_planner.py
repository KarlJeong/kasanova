"""쿼리 분해 플래너.

복잡한 멀티 조건 쿼리("A하고 B하고 C한 사용자")를 단일 SQL로 한 번에
생성하면 LLM이 JOIN 경로를 단순화해 거짓 양성을 만드는 경향이 있다
(예: "청약 = 보유"로 혼동). 이를 막기 위해 질문을 원자적 sub-question
으로 분해한 뒤, 각각을 기존 text_to_sql 파이프라인으로 독립 실행해
결과를 코드가 set 연산으로 결합하는 방식을 쓴다.

이 모듈은 분해 자체만 담당한다. 실행은 text_to_sql_tool이 수행.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

CombineOp = Literal["intersect", "union", "difference"]


class QueryPlan(BaseModel):
    """LLM이 만드는 쿼리 실행 계획."""

    requires_decomposition: bool = Field(
        description="분해가 필요한 쿼리인지 여부"
    )
    reasoning: str = Field(
        default="",
        description="판단 이유 (로그용, 사람이 읽을 수 있게)",
    )
    combine: CombineOp | None = Field(
        default=None,
        description=(
            "결합 연산. intersect=교집합(AND), "
            "union=합집합(OR), difference=차집합(BUT NOT)"
        ),
    )
    subqueries: list[str] = Field(
        default_factory=list,
        description="원자적 sub-question들 (필터용). 한국어 자연어.",
    )
    enrichment_queries: list[str] = Field(
        default_factory=list,
        description=(
            "필터가 아닌 추가 데이터 조회용 sub-question들. "
            "최종 필터 결과에 보강 정보를 붙일 때 사용."
        ),
    )


_PLANNER_SYSTEM_PROMPT = """\
너는 사내 데이터 질문을 받아 단계별 실행 계획을 세우는 query planner다.
복잡한 멀티 조건 질문을 원자적 sub-question으로 분해해, 결과 행들을
공통 키(예: 회원 식별 컬럼, 종목 식별 컬럼)로 set 연산하여 결합할 수
있게 만든다. 실제 키 컬럼명은 SQL 생성 단계에서 자동으로 결정된다.

[분해해야 하는 경우]
- 같은 엔티티(예: 사용자, DABS)에 대해 **여러 독립 조건이 AND/OR/BUT NOT**으로 연결된 질문.
- 예: "A를 청약하고 B를 보유 중이며 C 배당 수령한 사용자" → AND(intersect)
- 예: "A 또는 B를 청약한 사용자" → OR(union)
- 예: "A는 청약했지만 B는 안 한 사용자" → BUT NOT(difference)

[분해하지 말아야 하는 경우]
- 단일 조건 질문. 예: "이번 달 신규 가입 회원 수"
- 같은 행 내부의 두 컬럼을 비교하는 조건. 예: "청약 좌수가 보유 좌수의 50% 이상인 사용자"
- 단순 aggregation. 예: "DABS별 거래 금액 합계"
- 정렬·상한만 다른 조회. 예: "거래 금액 상위 10명"

[분해 규칙]
- 각 sub-question은 **정확히 하나의 독립 조건만** 담아야 한다.
  복수 조건을 하나의 sub-question에 묶으면 안 된다.
- sub-question 본문은 한국어 자연어로 자연스럽게 쓰되, 결과로 어떤 키
  (회원 식별값, 종목 코드 등)를 반환해야 하는지 명시.
- 결합 키는 모든 sub-question에서 동일한 의미여야 한다
  (예: 모두 회원 식별값, 또는 모두 DABS 코드).
- combine은 질문의 논리 접속사로 결정: "그리고/하고/이며/이면서" → intersect,
  "또는/이거나" → union, "했지만 ~ 안 한" → difference.
- subqueries는 최소 2개, 최대 5개.

[원자적 분해 — 매우 중요]
  원 질문이 N개의 독립 조건을 가지면 반드시 N개의 sub-question으로 분해하라.
  공통 전제(예: "공유증권 멤버 중")를 다른 조건과 묶어 하나의 sub-question으로
  합치면 안 된다 — 공통 전제도 독립 sub-question이다.

  잘못된 분해 (조건 합침 — 절대 금지):
      원 질문: "공유증권 멤버 중 마케팅 동의하고 특정 DABS 보유한 회원"
      X ["공유증권 멤버 중 마케팅 동의한 회원 목록",
          "공유증권 멤버 중 특정 DABS 보유 회원 목록"]
      → "공유증권 멤버" 조건이 양쪽에 중복. 각 SQL이 이를
         서로 다르게 해석할 위험이 있다.

  올바른 분해:
      원 질문: "공유증권 멤버 중 마케팅 동의하고 특정 DABS 보유한 회원"
      O ["공유증권 멤버인 회원 목록",
          "마케팅 동의한 회원 목록",
          "특정 DABS를 보유하고 있는 회원 목록"]
      → 세 조건 각각이 독립 sub-question. intersect로 결합.

[sub-question 작성 규칙 — 매우 중요]
- 각 sub-question은 **단독으로 읽어도 원 질문의 의도와 필터 범위가
  그대로 유지**되어야 한다. 다른 sub-question을 참조하지 않고도 혼자
  실행 가능해야 한다.
- 원 질문이 특정 DABS/종목/건물/기간/회원 유형 등 **특정 sub-question에
  속하는 제약**을 포함하면, 해당 sub-question 본문에 명시하라.
  literals 파라미터로 값이 따로 전달되더라도 sub-question 본문에
  해당 대상(예: DABS명 또는 코드)을 자연어로 다시 적어야
  SQL 생성기가 그 literal을 올바른 WHERE 조건에 바인딩할 수 있다.
- 원 질문이 "A에 ~한 사람 중 A에 ~하지 않은 사람"처럼 **같은 대상(A)을
  공유**하면, 두 sub-question 모두에 반드시 A를 명시하라.

  잘못된 분해 (공유 제약 누락 — 절대 금지):
      원 질문: "역삼 한국빌딩에 거래 이력이 없지만 현재 그 DABS를 보유 중인 멤버"
      X ["역삼 한국빌딩을 현재 보유 중인 회원 목록",
          "거래 이력이 있는 회원 목록"]
      → 두 번째 sub-question에서 "역삼 한국빌딩"이 사라져
         전체 거래자를 조회하게 된다. combine=difference로 결합하면
         정답 집합이 크게 오염된다.

  올바른 분해:
      원 질문: "역삼 한국빌딩에 거래 이력이 없지만 현재 그 DABS를 보유 중인 멤버"
      O ["역삼 한국빌딩을 현재 보유하고 있는 회원 목록",
          "역삼 한국빌딩에 거래 이력이 있는 회원 목록"]
      → 두 sub-question 모두 "역삼 한국빌딩"을 본문에 반복.
         combine=difference 로 결합하면 올바른 정답 집합이 된다.

[enrichment_queries — 보강 조회]
원 질문이 필터 조건 외에 **추가로 조회해야 할 데이터**(예: "의결권수를
알려줘", "각 회원의 연락처도 포함해줘")를 요구하면, 해당 조회를
enrichment_queries에 별도 sub-question으로 분리한다.

- enrichment_queries는 **필터가 아니라 데이터 조회**다.
  set 연산(intersect/union/difference)에 참여하지 않는다.
- 필터 subqueries의 결합 결과(최종 키 집합)에 대해 추가 정보를 붙이는
  용도이므로, enrichment sub-question도 반드시 동일한 키(회원 식별값
  등)를 반환해야 한다.
- enrichment sub-question 본문에도 대상 DABS/종목/건물 등 컨텍스트를
  명시해야 한다 (필터 sub-question과 동일한 규칙).
- 원 질문이 추가 데이터 조회를 요구하지 않으면 enrichment_queries는
  빈 배열로 둔다.

예시:
  원 질문: "그레인바운더리빌딩의 매각 투표가 가능한 회원들 중 마케팅
  수신 동의를 했고 아직 투표를 하지 않은 회원을 조회하고 각 회원이
  행사할 수 있는 의결권수를 알려줘"

  subqueries (필터용):
  - "그레인바운더리빌딩의 매각 투표가 가능한 회원 목록 (회원 식별값)"
  - "마케팅 수신 동의를 한 회원 목록 (회원 식별값)"
  - "그레인바운더리빌딩 매각 투표에 아직 참여하지 않은 회원 목록 (회원 식별값)"

  enrichment_queries (보강용):
  - "그레인바운더리빌딩 매각 투표에서 각 회원이 행사할 수 있는 의결권수 (회원 식별값, 의결권수)"

[출력 형식 — 반드시 JSON 한 객체만]
{
  "requires_decomposition": true | false,
  "reasoning": "왜 이렇게 판단했는지 한 줄 (한국어)",
  "combine": "intersect" | "union" | "difference" | null,
  "subqueries": ["sub-question 1", "sub-question 2", ...],
  "enrichment_queries": ["enrichment sub-question 1", ...]
}

분해가 불필요하면 requires_decomposition=false로 두고 나머지는 null/빈배열로 둔다.
JSON 외 다른 텍스트, 설명, 코드펜스 출력 금지.
"""


_USER_PROMPT_TEMPLATE = """\
다음 질문을 분석해 실행 계획을 JSON으로 출력하라.

질문: {query}

식별 리터럴 (참고): {literals}
"""


_JSON_BLOCK_PATTERN = re.compile(
    r"\{[\s\S]*\}",
    re.MULTILINE,
)
_CODE_FENCE_PATTERN = re.compile(
    r"^```(?:json)?\s*\n?|\n?```\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_fences(text: str) -> str:
    return _CODE_FENCE_PATTERN.sub("", text).strip()


def _extract_json(text: str) -> str | None:
    """LLM 응답에서 JSON 객체 블록을 추출한다."""
    cleaned = _strip_fences(text)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    match = _JSON_BLOCK_PATTERN.search(cleaned)
    if match:
        return match.group(0)
    return None


async def plan_query(
    query: str,
    literals: list[str],
    llm: BaseChatModel,
) -> QueryPlan:
    """질문을 분석해 분해 가능 여부 및 sub-question을 산출한다.

    LLM 호출이 실패하거나 JSON이 깨지면 안전하게
    `requires_decomposition=False`인 plan을 반환한다 (= 기존 단일 SQL
    경로로 폴백).
    """
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        query=query,
        literals=", ".join(literals) if literals else "(없음)",
    )
    try:
        response = await llm.ainvoke(
            [
                ("system", _PLANNER_SYSTEM_PROMPT),
                ("human", user_prompt),
            ]
        )
    except Exception:
        logger.exception(
            "[query_planner] LLM 호출 실패 → 분해 없이 폴백"
        )
        return QueryPlan(
            requires_decomposition=False,
            reasoning="planner LLM 호출 실패",
        )

    raw = getattr(response, "content", response)
    raw_text = str(raw)
    logger.info("[query_planner] LLM 응답: %s", raw_text[:500])

    json_block = _extract_json(raw_text)
    if not json_block:
        logger.warning(
            "[query_planner] JSON 블록 추출 실패 → 폴백"
        )
        return QueryPlan(
            requires_decomposition=False,
            reasoning="JSON 추출 실패",
        )

    try:
        data = json.loads(json_block)
    except json.JSONDecodeError:
        logger.warning(
            "[query_planner] JSON 파싱 실패: %s → 폴백",
            json_block[:200],
        )
        return QueryPlan(
            requires_decomposition=False,
            reasoning="JSON 파싱 실패",
        )

    try:
        plan = QueryPlan(**data)
    except ValidationError as e:
        logger.warning(
            "[query_planner] QueryPlan 검증 실패: %s → 폴백",
            e,
        )
        return QueryPlan(
            requires_decomposition=False,
            reasoning="plan 검증 실패",
        )

    # 분해 결정 시 필수 필드 검증 (LLM이 일관성 잃은 경우)
    if plan.requires_decomposition:
        if (
            not plan.subqueries
            or len(plan.subqueries) < 2
            or plan.combine is None
        ):
            logger.warning(
                "[query_planner] 분해 결정했으나 필드 부족"
                " (subqueries=%d, combine=%s) → 폴백",
                len(plan.subqueries),
                plan.combine,
            )
            return QueryPlan(
                requires_decomposition=False,
                reasoning="필수 필드 누락",
            )

    logger.info(
        "[query_planner] decomposition=%s reason=%s"
        " combine=%s sub_count=%d enrichment_count=%d",
        plan.requires_decomposition,
        plan.reasoning,
        plan.combine,
        len(plan.subqueries),
        len(plan.enrichment_queries),
    )
    if plan.requires_decomposition:
        for i, sub in enumerate(plan.subqueries, 1):
            logger.info(
                "[query_planner]   └ sub[%d/%d]: %s",
                i,
                len(plan.subqueries),
                sub,
            )
        for i, eq in enumerate(plan.enrichment_queries, 1):
            logger.info(
                "[query_planner]   └ enrichment[%d/%d]: %s",
                i,
                len(plan.enrichment_queries),
                eq,
            )

    return plan


def combine_keys(
    sets: list[set],
    combine: CombineOp,
) -> set:
    """sub-query 결과 키 집합을 결합 연산으로 합친다."""
    if not sets:
        return set()
    if combine == "intersect":
        return set.intersection(*sets)
    if combine == "union":
        return set.union(*sets)
    if combine == "difference":
        if len(sets) < 2:
            return sets[0]
        return sets[0] - set.union(*sets[1:])
    return set()
