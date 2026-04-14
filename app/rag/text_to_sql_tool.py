"""Text-to-SQL LangGraph tool (얇은 wrapper).

실제 파이프라인 로직은 `text_to_sql_graph.py`의 LangGraph
StateGraph로 옮겨졌다. 이 모듈은 그 컴파일된 그래프를 LangChain
`@tool`로 감싸 호출자(상위 LLM)에게 노출하기만 한다.

테스트 backward compatibility를 위해 `_extract_kb_tables` 등 일부
헬퍼를 helper 모듈에서 re-export 한다.
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from app.db.mysql_client import MySQLClient
from app.rag.schema_searcher import SchemaSearcher
from app.rag.searcher import HybridSearcher
from app.rag.text_to_sql_graph import (
    TextToSqlDeps,
    TextToSqlState,
    build_text_to_sql_graph,
)
from app.rag.text_to_sql_helpers import CANNOT_ANSWER

# ── backward-compat re-exports (테스트 / 외부 import 용) ──────────────
from app.rag.text_to_sql_helpers import (  # noqa: F401
    extract_kb_tables as _extract_kb_tables,
)
from app.rag.text_to_sql_helpers import (  # noqa: F401
    format_rows_for_tool_result as _format_rows_for_tool_result,
)

logger = logging.getLogger(__name__)


def create_text_to_sql_tool(
    schema_searcher: SchemaSearcher,
    kb_searcher: HybridSearcher,
    mysql_client: MySQLClient,
    llm: BaseChatModel,
) -> BaseTool:
    """Text-to-SQL LangGraph tool을 생성한다.

    내부적으로는 `text_to_sql_graph`의 컴파일된 StateGraph를 호출한다.
    그래프 내부의 노드 분할·상태 채널은 helpers 모듈과 graph 모듈에서
    관리한다.
    """
    deps = TextToSqlDeps(
        schema_searcher=schema_searcher,
        kb_searcher=kb_searcher,
        mysql_client=mysql_client,
        llm=llm,
    )
    graph = build_text_to_sql_graph(deps)

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
        - 다음 식별용 값은 query에 넣지 말고 literals로 분리하라
          (RAG 검색 품질을 떨어뜨리기 때문):
            · DABS 종목코드 (예: "KR011A200005X8")
            · DABS/종목/건물 고유명 (예: "북촌 월하재", "역삼 한국빌딩")
            · 회원 ID·이메일·사람 이름
            · 주소
            · 구체적 숫자·금액·날짜
        - 한국어 질문은 한국어 그대로. SQL·영어 의역 금지.
        - 이 도구 내부에서 스키마 검색 후 LLM이 SQL을 생성한다.
          호출자(상위 LLM)가 SQL을 만들면 안 된다.

        literals (선택):
        - SQL의 WHERE 절 등에 그대로 들어갈 식별용 값들의 리스트.
        - 예: DABS 종목코드, 종목 고유명, 회원 이메일, 특정 날짜 등.
        - 없으면 생략하거나 빈 리스트를 넘겨라.

        올바른 사용 예시:
        - query="이번 달 신규 가입 회원 수는?"
        - query="이 종목을 청약한 멤버 목록을 조회해줘",
          literals=["KR011A200005X8", "북촌 월하재"]
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

        initial_state: TextToSqlState = {
            "query": query,
            "literals": literals,
        }
        final_state = await graph.ainvoke(initial_state)
        return final_state.get("final_result", CANNOT_ANSWER)

    return text_to_sql_tool
