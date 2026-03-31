"""평가셋 자동 생성 — OpenSearch 청크 샘플링 + Claude Q&A 생성."""

import json
import logging
import os
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"

_QA_PROMPT = """아래 사내 문서 청크를 읽고, 이 내용을 기반으로 직원이 물어볼 법한 \
질문 1개와 그에 대한 정확한 답변 1개를 생성하세요.

## 문서 청크
{content}

## 출력 형식 (JSON만 반환)
{{"question": "...", "answer": "..."}}
"""


async def sample_chunks_from_opensearch(
    os_client: Any,
    index_name: str,
    category: str,
    n: int = 20,
) -> list[dict[str, Any]]:
    """카테고리별로 n개의 청크를 랜덤 샘플링한다."""
    body: dict[str, Any] = {
        "size": n,
        "query": {
            "function_score": {
                "query": {
                    "term": {"metadata.category": category}
                },
                "random_score": {},
            }
        },
    }

    response = await os_client.search(
        index=index_name, body=body
    )

    return [
        {
            "doc_id": hit["_source"]["doc_id"],
            "chunk_index": hit["_source"]["chunk_index"],
            "content": hit["_source"]["content"],
            "metadata": hit["_source"]["metadata"],
        }
        for hit in response["hits"]["hits"]
    ]


def generate_qa_pair(
    client: anthropic.Anthropic,
    chunk: dict[str, Any],
) -> dict[str, str]:
    """Claude Sonnet으로 청크에서 Q&A 쌍을 생성한다."""
    response = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": _QA_PROMPT.format(
                    content=chunk["content"]
                ),
            }
        ],
    )

    raw = response.content[0].text
    parsed = json.loads(raw)

    return {
        "question": parsed["question"],
        "ground_truth_answer": parsed["answer"],
    }


async def generate_dataset(
    os_client: Any,
    anthropic_client: anthropic.Anthropic,
    output_path: Path,
    n_per_category: int = 20,
    regenerate: bool = False,
) -> list[dict[str, Any]]:
    """평가셋을 생성하고 JSON 파일에 저장한다."""
    if output_path.exists() and not regenerate:
        logger.info(
            "기존 평가셋 사용: %s", output_path
        )
        return json.loads(output_path.read_text())

    categories = ["hr", "ops"]
    dataset: list[dict[str, Any]] = []

    for category in categories:
        chunks = await sample_chunks_from_opensearch(
            os_client,
            "kasanova_docs",
            category,
            n=n_per_category,
        )
        logger.info(
            "[%s] %d개 청크 샘플링 완료",
            category,
            len(chunks),
        )

        for chunk in chunks:
            qa = generate_qa_pair(
                anthropic_client, chunk
            )
            dataset.append(
                {
                    "question": qa["question"],
                    "ground_truth_answer": (
                        qa["ground_truth_answer"]
                    ),
                    "ground_truth_doc_ids": [
                        chunk["doc_id"]
                    ],
                    "category": category,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2)
    )
    logger.info(
        "평가셋 저장 완료: %s (%d건)",
        output_path,
        len(dataset),
    )
    return dataset
