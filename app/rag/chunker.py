import re
from dataclasses import dataclass


@dataclass
class Chunk:
    content: str
    chunk_index: int


def _count_tokens(text: str) -> int:
    """문자 수 기반으로 토큰 수를 근사한다 (1토큰 ≈ 3자)."""
    return max(1, len(text) // 3)


def _split_by_paragraphs(text: str) -> list[str]:
    """이중 개행으로 문단을 분리한다."""
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_sentences(text: str) -> list[str]:
    """문장 단위로 분리한다."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_words(text: str) -> list[str]:
    """단어 단위로 분리한다."""
    return text.split()


def _merge_segments(
    segments: list[str],
    max_tokens: int,
    overlap: int,
) -> list[str]:
    """세그먼트들을 max_tokens 이하의 청크로 병합한다."""
    chunks: list[str] = []
    current_words: list[str] = []

    for segment in segments:
        seg_words = segment.split()
        if not seg_words:
            continue

        if not current_words:
            current_words = seg_words
            continue

        combined_count = len(current_words) + len(seg_words)
        if combined_count <= max_tokens:
            current_words.extend(seg_words)
        else:
            chunks.append(" ".join(current_words))
            overlap_words = (
                current_words[-overlap:] if overlap > 0 else []
            )
            current_words = overlap_words + seg_words

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _split_segment(
    text: str,
    max_tokens: int,
    overlap: int,
) -> list[str]:
    """단일 세그먼트를 max_tokens 이하로 재분할한다."""
    if _count_tokens(text) <= max_tokens:
        return [text]

    sentences = _split_by_sentences(text)
    if len(sentences) > 1:
        result = _merge_segments(sentences, max_tokens, overlap)
        final: list[str] = []
        for r in result:
            if _count_tokens(r) <= max_tokens:
                final.append(r)
            else:
                words = _split_by_words(r)
                final.extend(
                    _merge_segments(words, max_tokens, overlap)
                )
        return final

    words = _split_by_words(text)
    return _merge_segments(words, max_tokens, overlap)


def chunk_text(
    text: str,
    max_tokens: int = 512,
    overlap: int = 50,
) -> list[Chunk]:
    """텍스트를 문단 > 문장 > 단어 순서로 분할하여 청크를 반환한다."""
    text = text.strip()
    if not text:
        return []

    if _count_tokens(text) <= max_tokens:
        return [Chunk(content=text, chunk_index=0)]

    paragraphs = _split_by_paragraphs(text)

    sub_segments: list[str] = []
    for para in paragraphs:
        if _count_tokens(para) <= max_tokens:
            sub_segments.append(para)
        else:
            sub_segments.extend(
                _split_segment(para, max_tokens, overlap)
            )

    raw_chunks = _merge_segments(
        sub_segments, max_tokens, overlap
    )

    final_chunks: list[str] = []
    for rc in raw_chunks:
        if _count_tokens(rc) <= max_tokens:
            final_chunks.append(rc)
        else:
            final_chunks.extend(
                _split_segment(rc, max_tokens, overlap)
            )

    return [
        Chunk(content=c, chunk_index=i)
        for i, c in enumerate(final_chunks)
    ]
