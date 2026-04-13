from unittest.mock import AsyncMock, MagicMock

import fitz
import pytest
from httpx import ASGITransport, AsyncClient

from app.rag.indexer import DocumentNotFoundError
from app.rag.parsers import UnsupportedFormatError


@pytest.fixture
def mock_indexer() -> MagicMock:
    indexer = MagicMock()
    indexer.index_document = AsyncMock(
        return_value={
            "doc_id": "report_abc12345",
            "filename": "report.pdf",
            "chunks": 42,
            "status": "indexed",
        }
    )
    indexer.list_documents = AsyncMock(
        return_value=[
            {
                "doc_id": "report_abc12345",
                "filename": "report.pdf",
                "chunks": 42,
                "indexed_at": "2026-03-26T12:00:00Z",
            }
        ]
    )
    indexer.delete_document = AsyncMock(return_value=None)
    return indexer


@pytest.fixture
def mock_schema_indexer() -> MagicMock:
    indexer = MagicMock()
    indexer.index_document = AsyncMock(
        return_value={
            "doc_id": "schema_abc12345",
            "filename": "schema.pdf",
            "chunks": 7,
            "status": "indexed",
        }
    )
    indexer.list_documents = AsyncMock(
        return_value=[
            {
                "doc_id": "schema_abc12345",
                "filename": "schema.pdf",
                "chunks": 7,
                "indexed_at": "2026-03-26T12:00:00Z",
            }
        ]
    )
    indexer.delete_document = AsyncMock(return_value=None)
    return indexer


@pytest.fixture
async def client(
    mock_indexer: MagicMock,
    mock_schema_indexer: MagicMock,
):
    from fastapi import FastAPI

    from app.api.index_router import router

    app = FastAPI()
    app.include_router(router)
    app.state.indexer = mock_indexer
    app.state.schema_indexer = mock_schema_indexer

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Test PDF content")
    data = doc.tobytes()
    doc.close()
    return data


class TestUploadDocument:
    async def test_upload_pdf_returns_indexed(
        self, client: AsyncClient, pdf_bytes: bytes
    ) -> None:
        resp = await client.post(
            "/index/documents",
            files={"file": ("report.pdf", pdf_bytes)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == "report_abc12345"
        assert data["status"] == "indexed"

    async def test_unsupported_format_returns_400(
        self,
        client: AsyncClient,
        mock_indexer: MagicMock,
    ) -> None:
        mock_indexer.index_document = AsyncMock(
            side_effect=UnsupportedFormatError("txt")
        )
        resp = await client.post(
            "/index/documents",
            files={"file": ("readme.txt", b"hello")},
        )
        assert resp.status_code == 400

    async def test_parse_failure_returns_422(
        self,
        client: AsyncClient,
        mock_indexer: MagicMock,
    ) -> None:
        mock_indexer.index_document = AsyncMock(
            side_effect=ValueError("파싱 실패")
        )
        resp = await client.post(
            "/index/documents",
            files={"file": ("bad.pdf", b"corrupt")},
        )
        assert resp.status_code == 422


class TestListDocuments:
    async def test_list_returns_documents(
        self, client: AsyncClient
    ) -> None:
        resp = await client.get("/index/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["documents"]) == 1
        assert (
            data["documents"][0]["doc_id"]
            == "report_abc12345"
        )


class TestDeleteDocument:
    async def test_delete_success(
        self, client: AsyncClient
    ) -> None:
        resp = await client.delete(
            "/index/documents/report_abc12345"
        )
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    async def test_delete_nonexistent_returns_404(
        self,
        client: AsyncClient,
        mock_indexer: MagicMock,
    ) -> None:
        mock_indexer.delete_document = AsyncMock(
            side_effect=DocumentNotFoundError("not found")
        )
        resp = await client.delete(
            "/index/documents/nonexistent"
        )
        assert resp.status_code == 404


class TestSchemaEndpoints:
    async def test_upload_routes_to_schema_indexer(
        self,
        client: AsyncClient,
        mock_indexer: MagicMock,
        mock_schema_indexer: MagicMock,
        pdf_bytes: bytes,
    ) -> None:
        resp = await client.post(
            "/index/schema/documents",
            files={"file": ("schema.pdf", pdf_bytes)},
        )
        assert resp.status_code == 200
        assert resp.json()["doc_id"] == "schema_abc12345"
        mock_schema_indexer.index_document.assert_awaited_once()
        mock_indexer.index_document.assert_not_awaited()

    async def test_upload_default_category_is_schema(
        self,
        client: AsyncClient,
        mock_schema_indexer: MagicMock,
        pdf_bytes: bytes,
    ) -> None:
        await client.post(
            "/index/schema/documents",
            files={"file": ("schema.pdf", pdf_bytes)},
        )
        kwargs = (
            mock_schema_indexer.index_document.call_args.kwargs
        )
        assert kwargs["category"] == "schema"

    async def test_list_returns_schema_documents(
        self,
        client: AsyncClient,
        mock_indexer: MagicMock,
    ) -> None:
        resp = await client.get("/index/schema/documents")
        assert resp.status_code == 200
        docs = resp.json()["documents"]
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "schema_abc12345"
        mock_indexer.list_documents.assert_not_awaited()

    async def test_delete_schema_document(
        self,
        client: AsyncClient,
        mock_schema_indexer: MagicMock,
    ) -> None:
        resp = await client.delete(
            "/index/schema/documents/schema_abc12345"
        )
        assert resp.status_code == 200
        mock_schema_indexer.delete_document.assert_awaited_once_with(
            "schema_abc12345"
        )

    async def test_delete_schema_nonexistent_returns_404(
        self,
        client: AsyncClient,
        mock_schema_indexer: MagicMock,
    ) -> None:
        mock_schema_indexer.delete_document = AsyncMock(
            side_effect=DocumentNotFoundError("not found")
        )
        resp = await client.delete(
            "/index/schema/documents/nonexistent"
        )
        assert resp.status_code == 404
