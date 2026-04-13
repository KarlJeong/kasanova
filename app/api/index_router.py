import logging
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from app.rag.indexer import (
    DocumentIndexer,
    DocumentNotFoundError,
)
from app.rag.parsers import UnsupportedFormatError

router = APIRouter(prefix="/index")


def _get_indexer(request: Request) -> DocumentIndexer:
    return request.app.state.indexer


def _get_schema_indexer(request: Request) -> DocumentIndexer:
    return request.app.state.schema_indexer


async def _upload(
    indexer: DocumentIndexer,
    file: UploadFile,
    category: str | None,
) -> dict:
    content = await file.read()
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    file_type = ext.lstrip(".")

    try:
        return await indexer.index_document(
            filename=filename,
            file_type=file_type,
            content=content,
            category=category,
        )
    except UnsupportedFormatError:
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 파일 형식입니다.",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"파일 파싱 실패: {e}",
        )
    except Exception as e:
        logging.getLogger(__name__).exception("인덱싱 실패")
        raise HTTPException(
            status_code=500,
            detail=f"인덱싱 실패: {e}",
        )


async def _list(indexer: DocumentIndexer) -> dict:
    documents = await indexer.list_documents()
    return {"documents": documents}


async def _delete(
    indexer: DocumentIndexer, doc_id: str
) -> dict:
    try:
        await indexer.delete_document(doc_id)
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="문서를 찾을 수 없습니다.",
        )
    return {"deleted": True}


@router.post("/documents")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    category: str | None = Form(None),
) -> dict:
    """문서를 업로드하고 기본 인덱스에 저장한다."""
    return await _upload(_get_indexer(request), file, category)


@router.get("/documents")
async def list_documents(request: Request) -> dict:
    """기본 인덱스의 문서 목록을 조회한다."""
    return await _list(_get_indexer(request))


@router.delete("/documents/{doc_id}")
async def delete_document(
    request: Request, doc_id: str
) -> dict:
    """기본 인덱스에서 문서를 삭제한다."""
    return await _delete(_get_indexer(request), doc_id)


@router.post("/schema/documents")
async def upload_schema_document(
    request: Request,
    file: UploadFile = File(...),
) -> dict:
    """문서를 스키마 인덱스(kasanova_schema)에 저장한다.

    .jsonl 파일은 각 행(table)을 독립 문서로 처리한다.
    그 외 확장자는 파일 단위로 인덱싱한다.
    """
    indexer = _get_schema_indexer(request)
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext == ".jsonl":
        content = await file.read()
        try:
            return await indexer.index_schema_jsonl(
                filename=filename,
                content=content,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail=f"JSONL 파싱 실패: {e}",
            )
        except Exception as e:
            logging.getLogger(__name__).exception(
                "JSONL 인덱싱 실패"
            )
            raise HTTPException(
                status_code=500,
                detail=f"인덱싱 실패: {e}",
            )

    return await _upload(indexer, file, category=None)


@router.get("/schema/documents")
async def list_schema_documents(request: Request) -> dict:
    """스키마 인덱스의 문서 목록을 조회한다."""
    return await _list(_get_schema_indexer(request))


@router.delete("/schema/documents/{doc_id}")
async def delete_schema_document(
    request: Request, doc_id: str
) -> dict:
    """스키마 인덱스에서 문서를 삭제한다."""
    return await _delete(
        _get_schema_indexer(request), doc_id
    )
