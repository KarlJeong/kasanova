"""동반 테이블 매핑 로더.

스키마 후보에 특정 테이블이 들어오면 SQL 생성 이전에 함께 주입할
테이블 목록을 YAML에서 읽어온다. 매핑은 데이터 폴더에 두어
스키마 YAML과 분리해 관리한다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_companion_tables(path: Path) -> dict[str, list[str]]:
    """YAML에서 `{trigger_table: [companion, ...]}` 매핑을 로드한다.

    파일이 없거나 `companions` 키가 비어 있으면 빈 dict를 반환한다.
    """
    if not path.exists():
        logger.info(
            "[companions] 매핑 파일 없음 — 주입 비활성: %s", path
        )
        return {}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    companions = raw.get("companions") or {}
    if not isinstance(companions, dict):
        raise ValueError(
            "table_companions.yaml의 companions 는 dict여야 합니다"
        )

    result: dict[str, list[str]] = {}
    for trigger, buddies in companions.items():
        if not buddies:
            continue
        if not isinstance(buddies, list):
            raise ValueError(
                f"companions[{trigger!r}] 값은 리스트여야 합니다"
            )
        result[trigger] = [str(b) for b in buddies]

    logger.info(
        "[companions] 매핑 %d건 로드: %s",
        len(result),
        list(result.keys()),
    )
    return result