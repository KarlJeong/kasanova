from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.encode.return_value = [[0.1] * 1024]
    return embedder


def _make_os_hit(
    doc_id: str = "doc_001",
    chunk_index: int = 0,
    content: str = "테스트 내용입니다.",
    filename: str = "test.pdf",
    category: str = "hr",
    score: float = 0.95,
) -> dict:
    return {
        "_score": score,
        "_source": {
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "content": content,
            "metadata": {
                "filename": filename,
                "category": category,
            },
        },
    }


def _make_mock_os_client(
    hits: list[dict] | None = None,
) -> AsyncMock:
    client = AsyncMock()
    client.search = AsyncMock(
        return_value={
            "hits": {
                "total": {"value": len(hits or [])},
                "hits": hits or [],
            }
        }
    )
    return client


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "doc_id": f"doc_{i:03d}",
            "chunk_index": 0,
            "content": f"청크 내용 {i}",
            "metadata": {
                "filename": f"file_{i}.pdf",
                "category": "hr" if i < 2 else "ops",
            },
        }
        for i in range(3)
    ]


@pytest.fixture
def sample_dataset() -> list[dict]:
    return [
        {
            "question": "연차 신청 절차는?",
            "ground_truth_answer": "연차는 시스템에서 신청합니다.",
            "ground_truth_doc_ids": ["doc_001"],
            "category": "hr",
        },
        {
            "question": "복리후생 항목은?",
            "ground_truth_answer": "식대, 교통비 등이 있습니다.",
            "ground_truth_doc_ids": ["doc_002"],
            "category": "hr",
        },
        {
            "question": "서비스 장애 대응 절차는?",
            "ground_truth_answer": "장애 발생 시 슬랙 채널에 공유합니다.",
            "ground_truth_doc_ids": ["doc_003"],
            "category": "ops",
        },
    ]


@pytest.fixture
def mock_os_client() -> AsyncMock:
    return _make_mock_os_client()


@pytest.fixture
def mock_embedder() -> MagicMock:
    return _make_mock_embedder()


@pytest.fixture
def tmp_dataset_path(tmp_path: Path) -> Path:
    return tmp_path / "eval_dataset.json"


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    return d
