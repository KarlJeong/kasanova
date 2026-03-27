import fitz


def parse_pdf(data: bytes) -> str:
    """PDF 바이트에서 페이지별 텍스트를 추출한다."""
    if not data:
        raise ValueError("빈 PDF 데이터입니다.")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ValueError(f"PDF 파싱 실패: {e}") from e

    pages: list[str] = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages.append(text)
    doc.close()

    if not pages:
        raise ValueError("PDF에서 텍스트를 추출할 수 없습니다.")
    return "\n\n".join(pages)
