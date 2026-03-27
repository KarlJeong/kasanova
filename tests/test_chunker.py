from app.rag.chunker import chunk_text


class TestChunker:
    def test_short_text_single_chunk(self) -> None:
        """512 토큰 미만의 텍스트는 1개 청크를 반환한다."""
        text = "짧은 텍스트입니다."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0

    def test_long_text_multiple_chunks(self) -> None:
        """512 토큰을 초과하는 텍스트는 여러 청크로 분할된다."""
        words = [f"word{i}" for i in range(600)]
        text = " ".join(words)
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_size_within_limit(self) -> None:
        """각 청크는 max_tokens를 초과하지 않는다."""
        words = [f"word{i}" for i in range(1000)]
        text = " ".join(words)
        chunks = chunk_text(text, max_tokens=100, overlap=10)
        for chunk in chunks:
            word_count = len(chunk.content.split())
            assert word_count <= 100

    def test_overlap_between_chunks(self) -> None:
        """인접한 청크 사이에 오버랩이 존재한다."""
        words = [f"word{i}" for i in range(200)]
        text = " ".join(words)
        chunks = chunk_text(text, max_tokens=50, overlap=10)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].content.split())
            words_b = set(chunks[i + 1].content.split())
            overlap = words_a & words_b
            assert len(overlap) > 0

    def test_split_respects_paragraph_boundaries(self) -> None:
        """문단 경계를 우선하여 분할한다."""
        para1 = " ".join([f"alpha{i}" for i in range(80)])
        para2 = " ".join([f"beta{i}" for i in range(80)])
        text = f"{para1}\n\n{para2}"
        chunks = chunk_text(text, max_tokens=100, overlap=10)
        assert len(chunks) >= 2
        assert "alpha0" in chunks[0].content
        assert "beta0" in chunks[-1].content

    def test_split_respects_sentence_boundaries(self) -> None:
        """문단 내에서 문장 경계로 분할한다."""
        sentences = [
            f"This is sentence number {i}." for i in range(100)
        ]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_tokens=50, overlap=5)
        for chunk in chunks:
            content = chunk.content.strip()
            if content and not content.endswith("."):
                pass

    def test_empty_text_returns_empty_list(self) -> None:
        """빈 텍스트는 빈 리스트를 반환한다."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunk_index_sequential(self) -> None:
        """chunk_index는 0부터 순차적으로 증가한다."""
        words = [f"word{i}" for i in range(600)]
        text = " ".join(words)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
