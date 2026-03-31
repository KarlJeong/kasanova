"""RAGAS 평가 — LangGraph 워크플로우 end-to-end 평가."""

import logging
import uuid
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.graph.nodes import call_llm
from app.graph.state import KasaNovaState
from app.rag.embedder import Embedder
from app.rag.retrieval_tool import create_retrieval_tool
from app.rag.searcher import HybridSearcher

logger = logging.getLogger(__name__)


class KureEmbeddings(Embeddings):
    """RAGAS용 KURE-v1 임베딩 어댑터."""

    def __init__(self) -> None:
        self._embedder = Embedder()

    def embed_documents(
        self, texts: list[str]
    ) -> list[list[float]]:
        return self._embedder.encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode([text])[0]


def build_eval_workflow(
    llm: BaseChatModel,
    searcher: HybridSearcher,
) -> Any:
    """평가용 워크플로우 — retrieval_tool만, MemorySaver."""
    retrieval_tool = create_retrieval_tool(searcher)
    tools = [retrieval_tool]
    llm_with_tools = llm.bind_tools(tools)
    llm_base = llm
    tool_node = ToolNode(tools)

    async def _call_llm(state: KasaNovaState) -> dict:
        return await call_llm(
            state, llm_with_tools, llm_base
        )

    async def _tool_node(state: KasaNovaState) -> dict:
        return await tool_node.ainvoke(state)

    builder = StateGraph(KasaNovaState)
    builder.add_node("call_llm", _call_llm)
    builder.add_node("tools", _tool_node)

    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges(
        "call_llm", tools_condition
    )
    builder.add_edge("tools", "call_llm")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


async def run_pipeline_item(
    workflow: Any,
    item: dict[str, Any],
) -> dict[str, Any]:
    """단일 평가 항목을 워크플로우로 실행한다."""
    from langchain_core.messages import HumanMessage

    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id}
    }

    result = await workflow.ainvoke(
        {"messages": [HumanMessage(content=item["question"])]},
        config=config,
    )

    messages = result["messages"]

    # 최종 답변 추출
    answer = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and not isinstance(
            msg, ToolMessage
        ):
            content = msg.content
            if isinstance(content, str) and content:
                answer = content
                break

    # 컨텍스트 추출 (ToolMessage)
    contexts = [
        msg.content
        for msg in messages
        if isinstance(msg, ToolMessage)
        and isinstance(msg.content, str)
    ]

    return {"answer": answer, "contexts": contexts}


def prepare_ragas_dataset(
    dataset: list[dict[str, Any]],
    pipeline_results: list[dict[str, Any]],
) -> EvaluationDataset:
    """평가 결과를 RAGAS EvaluationDataset으로 변환."""
    samples = []
    for item, result in zip(dataset, pipeline_results):
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=result["answer"],
                retrieved_contexts=result["contexts"],
                reference=item["ground_truth_answer"],
            )
        )
    return EvaluationDataset(samples=samples)


async def evaluate_pipeline(
    searcher: HybridSearcher,
    llm: BaseChatModel,
    judge_llm: BaseChatModel,
    dataset: list[dict[str, Any]],
    embeddings: Embeddings | None = None,
) -> dict[str, Any]:
    """LangGraph 워크플로우 + RAGAS 전체 평가."""
    workflow = build_eval_workflow(llm, searcher)

    # 각 항목 실행
    pipeline_results: list[dict[str, Any]] = []
    for i, item in enumerate(dataset):
        logger.info(
            "[RAGAS] [%d/%d] %s",
            i + 1,
            len(dataset),
            item["question"][:50],
        )
        result = await run_pipeline_item(
            workflow, item
        )
        pipeline_results.append(result)

    # RAGAS 평가
    eval_dataset = prepare_ragas_dataset(
        dataset, pipeline_results
    )

    metrics = [
        ContextPrecision(),
        ContextRecall(),
        Faithfulness(),
        AnswerRelevancy(),
    ]

    ragas_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
    )

    metric_keys = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]

    overall = {
        k: ragas_result[k]
        for k in metric_keys
        if k in ragas_result
    }

    # 카테고리별 분리
    by_category: dict[str, dict[str, float]] = {}
    cat_items: dict[str, list[int]] = {}
    for i, item in enumerate(dataset):
        cat = item["category"]
        if cat not in cat_items:
            cat_items[cat] = []
        cat_items[cat].append(i)

    # 카테고리별 RAGAS 평가
    for cat, indices in cat_items.items():
        cat_dataset_items = [dataset[i] for i in indices]
        cat_results = [
            pipeline_results[i] for i in indices
        ]
        cat_eval_dataset = prepare_ragas_dataset(
            cat_dataset_items, cat_results
        )
        cat_ragas_result = evaluate(
            dataset=cat_eval_dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=embeddings,
        )
        by_category[cat] = {
            k: cat_ragas_result[k]
            for k in metric_keys
            if k in cat_ragas_result
        }

    return {
        "overall": overall,
        "by_category": by_category,
    }
