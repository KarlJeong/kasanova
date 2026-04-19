import asyncio
import logging
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

logger = logging.getLogger(__name__)


def _resolve_model_path(model_name: str) -> Path:
    model_dir = _MODEL_DIR / model_name.replace("/", "--")
    if not model_dir.exists():
        raise RuntimeError(
            f"모델이 존재하지 않습니다: {model_dir}\n"
            f"다음 명령어로 모델을 다운로드하세요:\n"
            f"  huggingface-cli download {model_name}"
            f" --local-dir {model_dir}"
        )
    return model_dir


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Reranker:
    """bge-reranker-v2-m3 CrossEncoder로
    (query, passage) 쌍의 관련도를 평가한다."""

    def __init__(
        self, model_name: str = _DEFAULT_MODEL
    ) -> None:
        path = _resolve_model_path(model_name)
        device = _select_device()
        self.model = CrossEncoder(
            str(path), device=device
        )
        logger.info(
            "Reranker 디바이스: %s", device
        )

    def rank(
        self, query: str, passages: list[str]
    ) -> list[float]:
        """각 passage의 관련도 점수를 반환한다."""
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs)
        return scores.tolist()

    async def arank(
        self, query: str, passages: list[str]
    ) -> list[float]:
        """비동기 래퍼."""
        return await asyncio.to_thread(
            self.rank, query, passages
        )
