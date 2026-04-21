"""table_companions 로더 단위 테스트."""
from pathlib import Path

import pytest

from app.rag.table_companions import load_companion_tables


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_companion_tables(tmp_path / "missing.yaml")
    assert result == {}


def test_empty_companions_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("companions: {}\n", encoding="utf-8")
    assert load_companion_tables(p) == {}


def test_loads_mapping(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "companions:\n"
        "  kasa_offering_subscription:\n"
        "    - kasa_offering\n"
        "  kasa_member:\n"
        "    - kasa_user_individual_member\n"
        "    - kasa_user_corporate_member\n",
        encoding="utf-8",
    )
    result = load_companion_tables(p)
    assert result == {
        "kasa_offering_subscription": ["kasa_offering"],
        "kasa_member": [
            "kasa_user_individual_member",
            "kasa_user_corporate_member",
        ],
    }


def test_invalid_list_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        "companions:\n  kasa_offering_subscription: kasa_offering\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_companion_tables(p)


def test_invalid_root_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("companions: [a, b]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_companion_tables(p)