from pathlib import Path

from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "nlpai-lab/KURE-v1"
_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def _resolve_model_path(model_name: str) -> Path:
    """프로젝트 models/ 디렉토리에서 모델 경로를 반환한다."""
    model_dir = _MODEL_DIR / model_name.replace("/", "--")
    if not model_dir.exists():
        raise RuntimeError(
            f"모델이 존재하지 않습니다: {model_dir}\n"
            f"다음 명령어로 모델을 다운로드하세요:\n"
            f"  huggingface-cli download {model_name}"
            f" --local-dir {model_dir}"
        )
    return model_dir


class Embedder:
    """KURE-v1 모델을 래핑하여 텍스트 임베딩을 제공한다."""

    def __init__(
        self, model_name: str = _DEFAULT_MODEL
    ) -> None:
        path = _resolve_model_path(model_name)
        self.model = SentenceTransformer(str(path))
        print("max_seq_length:", self.model.max_seq_length)

    def encode(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """텍스트 리스트를 임베딩 벡터 리스트로 변환한다."""
        embeddings = self.model.encode(
            texts, batch_size=batch_size
        )
        return embeddings.tolist()
