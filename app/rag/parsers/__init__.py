from collections.abc import Callable

from app.rag.parsers.pdf_parser import parse_pdf
from app.rag.parsers.pptx_parser import parse_pptx
from app.rag.parsers.docx_parser import parse_docx


class UnsupportedFormatError(Exception):
    """지원하지 않는 파일 형식일 때 발생한다."""


_PARSERS_BY_EXT: dict[str, Callable[[bytes], str]] = {
    ".pdf": parse_pdf,
    ".pptx": parse_pptx,
    ".docx": parse_docx,
}


def parse_by_extension(ext: str, data: bytes) -> str:
    """파일 확장자에 따라 적절한 파서를 선택하여 텍스트를 추출한다."""
    parser = _PARSERS_BY_EXT.get(ext.lower())
    if parser is None:
        raise UnsupportedFormatError(
            f"지원하지 않는 파일 형식입니다: {ext}"
        )
    return parser(data)
