import logging
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)

from app.rag.indexer import (
    DocumentIndexer,
    DocumentNotFoundError,
)
from app.rag.parsers import UnsupportedFormatError

router = APIRouter(prefix="/index")


def _resolve_indexer(
    request: Request, index_name: str | None
) -> DocumentIndexer:
    indexers: dict[str, DocumentIndexer] | None = getattr(
        request.app.state, "indexers", None
    )
    if not indexers:
        fallback = request.app.state.indexer
        indexers = {fallback.index_name: fallback}
    default_index = getattr(
        request.app.state,
        "default_index",
        request.app.state.indexer.index_name,
    )
    name = index_name or default_index
    if name not in indexers:
        raise HTTPException(
            status_code=400,
            detail=(
                f"허용되지 않은 인덱스입니다: {name}. "
                f"허용 목록: {sorted(indexers.keys())}"
            ),
        )
    return indexers[name]


@router.post("/documents")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    category: str = Form("general"),
    index_name: str | None = Form(None),
) -> dict:
    """문서를 업로드하고 인덱싱한다."""
    indexer = _resolve_indexer(request, index_name)
    content = await file.read()
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    file_type = ext.lstrip(".")

    try:
        result = await indexer.index_document(
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

    return result


@router.get("/documents")
async def list_documents(
    request: Request,
    index_name: str | None = Query(None),
) -> dict:
    """인덱싱된 문서 목록을 조회한다."""
    indexer = _resolve_indexer(request, index_name)
    documents = await indexer.list_documents()
    return {"documents": documents}


@router.delete("/documents/{doc_id}")
async def delete_document(
    request: Request,
    doc_id: str,
    index_name: str | None = Query(None),
) -> dict:
    """문서를 삭제한다."""
    indexer = _resolve_indexer(request, index_name)
    try:
        await indexer.delete_document(doc_id)
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="문서를 찾을 수 없습니다.",
        )
    return {"deleted": True}
