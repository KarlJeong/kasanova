import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.eval.conftest import _make_mock_os_client, _make_os_hit


class TestSampleChunks:
    async def test_sample_chunks_from_opensearch(self) -> None:
        from eval.generate_dataset import sample_chunks_from_opensearch

        hits = [
            _make_os_hit(
                doc_id=f"doc_{i:03d}", category="hr"
            )
            for i in range(3)
        ]
        os_client = _make_mock_os_client(hits)

        result = await sample_chunks_from_opensearch(
            os_client, "kasanova_docs", "hr", n=3
        )

        assert len(result) == 3
        assert result[0]["doc_id"] == "doc_000"
        assert result[0]["metadata"]["category"] == "hr"

        call_body = os_client.search.call_args[1]["body"]
        query = call_body["query"]
        assert "function_score" in query
        assert query["function_score"]["query"] == {
            "term": {"metadata.category": "hr"}
        }

    async def test_sample_returns_empty_on_no_hits(
        self,
    ) -> None:
        from eval.generate_dataset import sample_chunks_from_opensearch

        os_client = _make_mock_os_client(hits=[])
        result = await sample_chunks_from_opensearch(
            os_client, "kasanova_docs", "hr", n=5
        )
        assert result == []


class TestGenerateQAPair:
    def test_generate_qa_pair(self) -> None:
        from eval.generate_dataset import generate_qa_pair

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "question": "연차 신청은 어떻게 하나요?",
                        "answer": "시스템에서 신청합니다.",
                    },
                    ensure_ascii=False,
                )
            )
        ]
        mock_client.messages.create.return_value = (
            mock_response
        )

        chunk = {
            "doc_id": "doc_001",
            "content": "연차 신청 절차는 다음과 같습니다.",
            "metadata": {"category": "hr"},
        }

        result = generate_qa_pair(mock_client, chunk)

        assert "question" in result
        assert "ground_truth_answer" in result
        assert result["question"] == "연차 신청은 어떻게 하나요?"
        assert result["ground_truth_answer"] == "시스템에서 신청합니다."

        mock_client.messages.create.assert_called_once()
        call_kwargs = (
            mock_client.messages.create.call_args[1]
        )
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"


class TestGenerateDataset:
    async def test_creates_file(
        self,
        tmp_dataset_path: Path,
    ) -> None:
        from eval.generate_dataset import generate_dataset

        hr_hits = [
            _make_os_hit(
                doc_id=f"hr_{i}", category="hr",
                content=f"HR 내용 {i}",
            )
            for i in range(2)
        ]
        ops_hits = [
            _make_os_hit(
                doc_id=f"ops_{i}", category="ops",
                content=f"OPS 내용 {i}",
            )
            for i in range(2)
        ]

        os_client = AsyncMock()
        os_client.search = AsyncMock(
            side_effect=[
                {
                    "hits": {
                        "total": {"value": 2},
                        "hits": hr_hits,
                    }
                },
                {
                    "hits": {
                        "total": {"value": 2},
                        "hits": ops_hits,
                    }
                },
            ]
        )

        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "question": "테스트 질문",
                        "answer": "테스트 답변",
                    },
                    ensure_ascii=False,
                )
            )
        ]
        mock_anthropic.messages.create.return_value = (
            mock_response
        )

        result = await generate_dataset(
            os_client,
            mock_anthropic,
            tmp_dataset_path,
            n_per_category=2,
        )

        assert tmp_dataset_path.exists()
        assert len(result) == 4

        saved = json.loads(tmp_dataset_path.read_text())
        assert len(saved) == 4

    async def test_skips_if_exists(
        self,
        tmp_dataset_path: Path,
    ) -> None:
        from eval.generate_dataset import generate_dataset

        existing = [{"question": "기존 데이터"}]
        tmp_dataset_path.write_text(
            json.dumps(existing, ensure_ascii=False)
        )

        os_client = AsyncMock()
        mock_anthropic = MagicMock()

        result = await generate_dataset(
            os_client,
            mock_anthropic,
            tmp_dataset_path,
        )

        assert result == existing
        os_client.search.assert_not_called()

    async def test_regenerate_overwrites(
        self,
        tmp_dataset_path: Path,
    ) -> None:
        from eval.generate_dataset import generate_dataset

        existing = [{"question": "기존 데이터"}]
        tmp_dataset_path.write_text(
            json.dumps(existing, ensure_ascii=False)
        )

        hits = [
            _make_os_hit(
                doc_id=f"doc_{i}", content=f"내용 {i}"
            )
            for i in range(2)
        ]
        os_client = AsyncMock()
        os_client.search = AsyncMock(
            return_value={
                "hits": {
                    "total": {"value": 2},
                    "hits": hits,
                }
            }
        )

        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "question": "새 질문",
                        "answer": "새 답변",
                    },
                    ensure_ascii=False,
                )
            )
        ]
        mock_anthropic.messages.create.return_value = (
            mock_response
        )

        result = await generate_dataset(
            os_client,
            mock_anthropic,
            tmp_dataset_path,
            n_per_category=2,
            regenerate=True,
        )

        assert len(result) == 4
        saved = json.loads(tmp_dataset_path.read_text())
        assert saved[0]["question"] == "새 질문"

    def test_eval_item_schema(
        self, sample_dataset: list[dict]
    ) -> None:
        required_keys = {
            "question",
            "ground_truth_answer",
            "ground_truth_doc_ids",
            "category",
        }
        for item in sample_dataset:
            assert set(item.keys()) == required_keys
            assert isinstance(
                item["ground_truth_doc_ids"], list
            )
            assert all(
                isinstance(d, str)
                for d in item["ground_truth_doc_ids"]
            )
