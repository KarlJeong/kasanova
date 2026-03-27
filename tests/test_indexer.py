import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import fitz
import pytest

from app.rag.indexer import DocumentIndexer, DocumentNotFoundError


def _make_mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.encode.return_value = [[0.1] * 1024]
    return embedder


def _make_mock_os_client() -> AsyncMock:
    client = AsyncMock()
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock()
    client.delete_by_query = AsyncMock(
        return_value={"deleted": 0}
    )
    return client


@pytest.fixture
def pdf_content() -> bytes:
    """유효한 PDF 바이트를 생성한다."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Test document content")
    data = doc.tobytes()
    doc.close()
    return data


class TestIndexDocument:
    @pytest.fixture
    def indexer(self) -> DocumentIndexer:
        return DocumentIndexer(
            os_client=_make_mock_os_client(),
            embedder=_make_mock_embedder(),
            index_name="test_index",
        )

    async def test_returns_doc_metadata(
        self, indexer: DocumentIndexer, pdf_content: bytes
    ) -> None:
        file_hash = hashlib.sha256(pdf_content).hexdigest()[:8]

        with patch(
            "app.rag.indexer.async_bulk",
            new_callable=AsyncMock,
        ) as mock_bulk:
            mock_bulk.return_value = (1, [])
            result = await indexer.index_document(
                filename="report.pdf",
                file_type="pdf",
                content=pdf_content,
            )

        assert result["filename"] == "report.pdf"
        assert result["doc_id"] == f"report_{file_hash}"
        assert result["status"] == "indexed"
        assert result["chunks"] > 0

    async def test_calls_bulk_index(
        self, indexer: DocumentIndexer, pdf_content: bytes
    ) -> None:
        with patch(
            "app.rag.indexer.async_bulk",
            new_callable=AsyncMock,
        ) as mock_bulk:
            mock_bulk.return_value = (1, [])
            await indexer.index_document(
                filename="test.pdf",
                file_type="pdf",
                content=pdf_content,
            )

        mock_bulk.assert_called_once()
        args = mock_bulk.call_args
        actions = list(args[0][1])
        assert len(actions) > 0
        assert actions[0]["_index"] == "test_index"

    async def test_deletes_existing_before_reindex(
        self, indexer: DocumentIndexer, pdf_content: bytes
    ) -> None:
        indexer.os_client.delete_by_query = AsyncMock(
            return_value={"deleted": 5}
        )

        with patch(
            "app.rag.indexer.async_bulk",
            new_callable=AsyncMock,
        ) as mock_bulk:
            mock_bulk.return_value = (1, [])
            await indexer.index_document(
                filename="report.pdf",
                file_type="pdf",
                content=pdf_content,
            )

        indexer.os_client.delete_by_query.assert_called_once()
        call_kwargs = (
            indexer.os_client.delete_by_query.call_args
        )
        assert call_kwargs[1]["index"] == "test_index"

    async def test_embedding_called_with_chunks(
        self, indexer: DocumentIndexer, pdf_content: bytes
    ) -> None:
        indexer.embedder.encode.return_value = [[0.1] * 1024]

        with patch(
            "app.rag.indexer.async_bulk",
            new_callable=AsyncMock,
        ) as mock_bulk:
            mock_bulk.return_value = (1, [])
            await indexer.index_document(
                filename="test.pdf",
                file_type="pdf",
                content=pdf_content,
            )

        indexer.embedder.encode.assert_called_once()
        texts = indexer.embedder.encode.call_args[0][0]
        assert isinstance(texts, list)
        assert len(texts) > 0


class TestDeleteDocument:
    async def test_delete_existing_document(self) -> None:
        client = _make_mock_os_client()
        client.delete_by_query = AsyncMock(
            return_value={"deleted": 5}
        )
        indexer = DocumentIndexer(
            os_client=client,
            embedder=_make_mock_embedder(),
            index_name="test_index",
        )

        await indexer.delete_document("report_abc12345")

        client.delete_by_query.assert_called_once_with(
            index="test_index",
            body={
                "query": {
                    "term": {"doc_id": "report_abc12345"}
                }
            },
        )

    async def test_delete_nonexistent_raises(self) -> None:
        client = _make_mock_os_client()
        client.delete_by_query = AsyncMock(
            return_value={"deleted": 0}
        )
        indexer = DocumentIndexer(
            os_client=client,
            embedder=_make_mock_embedder(),
            index_name="test_index",
        )

        with pytest.raises(DocumentNotFoundError):
            await indexer.delete_document("nonexistent_id")


class TestListDocuments:
    async def test_returns_document_list(self) -> None:
        client = _make_mock_os_client()
        client.search = AsyncMock(
            return_value={
                "aggregations": {
                    "docs": {
                        "buckets": [
                            {
                                "key": "report_abc12345",
                                "doc_count": 42,
                                "latest": {
                                    "hits": {
                                        "hits": [
                                            {
                                                "_source": {
                                                    "metadata": {
                                                        "filename": "report.pdf",
                                                        "indexed_at": "2026-03-26T12:00:00Z",
                                                    }
                                                }
                                            }
                                        ]
                                    }
                                },
                            }
                        ]
                    }
                }
            }
        )
        indexer = DocumentIndexer(
            os_client=client,
            embedder=_make_mock_embedder(),
            index_name="test_index",
        )

        docs = await indexer.list_documents()

        assert len(docs) == 1
        assert docs[0]["doc_id"] == "report_abc12345"
        assert docs[0]["filename"] == "report.pdf"
        assert docs[0]["chunks"] == 42
        assert docs[0]["indexed_at"] == "2026-03-26T12:00:00Z"

    async def test_empty_index_returns_empty_list(
        self,
    ) -> None:
        client = _make_mock_os_client()
        client.search = AsyncMock(
            return_value={
                "aggregations": {"docs": {"buckets": []}}
            }
        )
        indexer = DocumentIndexer(
            os_client=client,
            embedder=_make_mock_embedder(),
            index_name="test_index",
        )

        docs = await indexer.list_documents()
        assert docs == []
