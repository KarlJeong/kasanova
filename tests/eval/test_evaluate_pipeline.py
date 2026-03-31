from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)


class TestBuildEvalWorkflow:
    def test_uses_memory_saver(self) -> None:
        from eval.evaluate_pipeline import (
            build_eval_workflow,
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_searcher = MagicMock()

        workflow = build_eval_workflow(
            mock_llm, mock_searcher
        )

        from langgraph.checkpoint.memory import (
            MemorySaver,
        )

        assert isinstance(
            workflow.checkpointer, MemorySaver
        )

    def test_only_retrieval_tool(self) -> None:
        from eval.evaluate_pipeline import (
            build_eval_workflow,
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_searcher = MagicMock()

        build_eval_workflow(mock_llm, mock_searcher)

        bind_call = mock_llm.bind_tools.call_args
        tools = bind_call[0][0]
        tool_names = [t.name for t in tools]
        assert "retrieval_tool" in tool_names
        assert "web_search" not in tool_names
        assert len(tools) == 1


class TestRunPipelineItem:
    async def test_returns_answer_and_contexts(
        self,
    ) -> None:
        from eval.evaluate_pipeline import (
            run_pipeline_item,
        )

        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = {
            "messages": [
                HumanMessage(content="질문"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc_1",
                            "name": "retrieval_tool",
                            "args": {"query": "질문"},
                        }
                    ],
                ),
                ToolMessage(
                    content="검색 결과 내용",
                    tool_call_id="tc_1",
                    name="retrieval_tool",
                ),
                AIMessage(content="최종 답변입니다."),
            ]
        }

        result = await run_pipeline_item(
            mock_workflow,
            {"question": "질문", "category": "hr"},
        )

        assert result["answer"] == "최종 답변입니다."
        assert "검색 결과 내용" in result["contexts"]

    async def test_extracts_multiple_contexts(
        self,
    ) -> None:
        from eval.evaluate_pipeline import (
            run_pipeline_item,
        )

        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = {
            "messages": [
                HumanMessage(content="질문"),
                ToolMessage(
                    content="첫 번째 결과",
                    tool_call_id="tc_1",
                    name="retrieval_tool",
                ),
                ToolMessage(
                    content="두 번째 결과",
                    tool_call_id="tc_2",
                    name="retrieval_tool",
                ),
                AIMessage(content="답변"),
            ]
        }

        result = await run_pipeline_item(
            mock_workflow,
            {"question": "질문", "category": "hr"},
        )

        assert len(result["contexts"]) == 2


class TestPrepareRagasDataset:
    def test_creates_evaluation_dataset(self) -> None:
        from eval.evaluate_pipeline import (
            prepare_ragas_dataset,
        )

        dataset = [
            {
                "question": "질문1",
                "ground_truth_answer": "정답1",
                "ground_truth_doc_ids": ["doc_1"],
                "category": "hr",
            },
        ]
        pipeline_results = [
            {
                "answer": "답변1",
                "contexts": ["컨텍스트1"],
            },
        ]

        eval_dataset = prepare_ragas_dataset(
            dataset, pipeline_results
        )

        from ragas import EvaluationDataset

        assert isinstance(eval_dataset, EvaluationDataset)
        assert len(eval_dataset) == 1

        sample = eval_dataset[0]
        assert sample.user_input == "질문1"
        assert sample.response == "답변1"
        assert sample.retrieved_contexts == ["컨텍스트1"]
        assert sample.reference == "정답1"


class TestEvaluatePipeline:
    @patch("eval.evaluate_pipeline.evaluate")
    async def test_returns_metrics(
        self,
        mock_ragas_evaluate: MagicMock,
        sample_dataset: list[dict],
    ) -> None:
        from eval.evaluate_pipeline import (
            evaluate_pipeline,
        )

        mock_scores = {
            "context_precision": 0.80,
            "context_recall": 0.75,
            "faithfulness": 0.90,
            "answer_relevancy": 0.85,
        }
        mock_result = MagicMock()
        mock_result.__getitem__ = (
            lambda self, key: mock_scores[key]
        )
        mock_result.__contains__ = (
            lambda self, key: key in mock_scores
        )
        mock_result.keys = lambda: mock_scores.keys()
        mock_ragas_evaluate.return_value = mock_result

        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = {
            "messages": [
                HumanMessage(content="q"),
                ToolMessage(
                    content="ctx",
                    tool_call_id="tc",
                    name="retrieval_tool",
                ),
                AIMessage(content="answer"),
            ]
        }

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_searcher = MagicMock()
        mock_judge = MagicMock()

        with patch(
            "eval.evaluate_pipeline.build_eval_workflow",
            return_value=mock_workflow,
        ):
            result = await evaluate_pipeline(
                mock_searcher,
                mock_llm,
                mock_judge,
                sample_dataset,
            )

        assert "overall" in result
        assert "by_category" in result
        assert (
            result["overall"]["context_precision"] == 0.80
        )
        assert result["overall"]["faithfulness"] == 0.90
