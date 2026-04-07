def parse_md(data: bytes) -> str:
    """Markdown 바이트에서 텍스트를 추출한다."""
    text = data.decode("utf-8").strip()
    if not text:
        raise ValueError("빈 Markdown 데이터입니다.")
    return text
