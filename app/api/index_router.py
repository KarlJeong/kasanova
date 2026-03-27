import logging
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
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


@router.post("/documents")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> dict:
    """문서를 업로드하고 인덱싱한다."""
    indexer = _get_indexer(request)
    content = await file.read()
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    file_type = ext.lstrip(".")

    try:
        result = await indexer.index_document(
            filename=filename,
            file_type=file_type,
            content=content,
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

    return result


@router.get("/documents")
async def list_documents(request: Request) -> dict:
    """인덱싱된 문서 목록을 조회한다."""
    indexer = _get_indexer(request)
    documents = await indexer.list_documents()
    return {"documents": documents}


@router.delete("/documents/{doc_id}")
async def delete_document(
    request: Request, doc_id: str
) -> dict:
    """문서를 삭제한다."""
    indexer = _get_indexer(request)
    try:
        await indexer.delete_document(doc_id)
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="문서를 찾을 수 없습니다.",
        )
    return {"deleted": True}
