import io

from docx import Document


def parse_docx(data: bytes) -> str:
    """DOCX 바이트에서 단락 단위 텍스트를 추출한다."""
    if not data:
        raise ValueError("빈 DOCX 데이터입니다.")
    try:
        doc = Document(io.BytesIO(data))
    except Exception as e:
        raise ValueError(f"DOCX 파싱 실패: {e}") from e

    paragraphs: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    if not paragraphs:
        raise ValueError("DOCX에서 텍스트를 추출할 수 없습니다.")
    return "\n\n".join(paragraphs)
