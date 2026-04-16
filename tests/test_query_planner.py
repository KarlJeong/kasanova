from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.rag.query_planner import (
    QueryPlan,
    combine_keys,
    plan_query,
)


@pytest.fixture
def mock_llm() -> MagicMock:
    return MagicMock()


class TestCombineKeys:
    def test_intersect_basic(self) -> None:
        assert combine_keys(
            [{1, 2, 3}, {2, 3, 4}, {3, 4, 5}], "intersect"
        ) == {3}

    def test_union_basic(self) -> None:
        assert combine_keys(
            [{1, 2}, {2, 3}, {3, 4}], "union"
        ) == {1, 2, 3, 4}

    def test_difference_basic(self) -> None:
        # 첫 set에서 나머지 set들의 합집합을 빼기
        assert combine_keys(
            [{1, 2, 3, 4}, {2}, {3}], "difference"
        ) == {1, 4}

    def test_empty_input(self) -> None:
        assert combine_keys([], "intersect") == set()

    def test_difference_single_set(self) -> None:
        # set 하나만 있으면 그대로 반환
        assert combine_keys([{1, 2}], "difference") == {1, 2}


class TestPlanQuery:
    async def test_decomposes_multi_condition(
        self, mock_llm
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content=(
                    '{"requires_decomposition": true,'
                    ' "reasoning": "세 조건 AND",'
                    ' '
                    ' "combine": "intersect",'
                    ' "subqueries": ['
                    '"sub1", "sub2", "sub3"]}'
                )
            )
        )
        plan = await plan_query(
            "A하고 B하고 C한 사용자",
            ["KR001"],
            mock_llm,
        )
        assert plan.requires_decomposition is True
        assert plan.combine == "intersect"
        assert len(plan.subqueries) == 3

    async def test_single_condition_not_decomposed(
        self, mock_llm
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content=(
                    '{"requires_decomposition": false,'
                    ' "reasoning": "단일 카운트 질문",'
                                        ' "combine": null,'
                    ' "subqueries": []}'
                )
            )
        )
        plan = await plan_query(
            "이번 달 신규 가입 회원 수",
            [],
            mock_llm,
        )
        assert plan.requires_decomposition is False

    async def test_strips_code_fence_and_parses(
        self, mock_llm
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content=(
                    '```json\n'
                    '{"requires_decomposition": false,'
                    ' "reasoning": "x",'
                                        ' "combine": null,'
                    ' "subqueries": []}\n'
                    '```'
                )
            )
        )
        plan = await plan_query("질문", [], mock_llm)
        assert plan.requires_decomposition is False
        assert plan.reasoning == "x"

    async def test_extracts_json_from_noisy_text(
        self, mock_llm
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content=(
                    "분석 결과:\n"
                    '{"requires_decomposition": true,'
                    ' "reasoning": "교집합",'
                    ' '
                    ' "combine": "intersect",'
                    ' "subqueries": ["a", "b"]}\n'
                    "이상입니다."
                )
            )
        )
        plan = await plan_query("질문", [], mock_llm)
        assert plan.requires_decomposition is True
        assert plan.combine == "intersect"

    async def test_invalid_json_falls_back(
        self, mock_llm
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="이건 JSON이 아니다"
            )
        )
        plan = await plan_query("질문", [], mock_llm)
        assert plan.requires_decomposition is False

    async def test_decomposition_without_subqueries_falls_back(
        self, mock_llm
    ) -> None:
        """LLM이 분해 결정했지만 subqueries 비어있으면 안전하게
        폴백."""
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content=(
                    '{"requires_decomposition": true,'
                    ' "reasoning": "x",'
                    ' '
                    ' "combine": "intersect",'
                    ' "subqueries": []}'
                )
            )
        )
        plan = await plan_query("질문", [], mock_llm)
        assert plan.requires_decomposition is False

    async def test_decomposition_missing_combine_falls_back(
        self, mock_llm
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content=(
                    '{"requires_decomposition": true,'
                    ' "reasoning": "x",'
                    ' '
                    ' "combine": null,'
                    ' "subqueries": ["a", "b"]}'
                )
            )
        )
        plan = await plan_query("질문", [], mock_llm)
        assert plan.requires_decomposition is False

    async def test_llm_exception_falls_back(
        self, mock_llm
    ) -> None:
        mock_llm.ainvoke = AsyncMock(
            side_effect=RuntimeError("LLM down")
        )
        plan = await plan_query("질문", [], mock_llm)
        assert plan.requires_decomposition is False
        assert "실패" in plan.reasoning


class TestQueryPlanModel:
    def test_default_no_decomposition(self) -> None:
        plan = QueryPlan(requires_decomposition=False)
        assert plan.combine is None
        assert plan.subqueries == []
