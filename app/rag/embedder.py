from sentence_transformers import SentenceTransformer


class Embedder:
    """KURE-v1 모델을 래핑하여 텍스트 임베딩을 제공한다."""

    def __init__(
        self, model_name: str = "nlpai-lab/KURE-v1"
    ) -> None:
        try:
            self.model = SentenceTransformer(
                model_name, local_files_only=True
            )
        except OSError:
            self.model = SentenceTransformer(model_name)

    def encode(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """텍스트 리스트를 임베딩 벡터 리스트로 변환한다."""
        embeddings = self.model.encode(
            texts, batch_size=batch_size
        )
        return embeddings.tolist()
