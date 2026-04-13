import json

import pytest

from app.rag.schema_jsonl import (
    parse_schema_jsonl,
    split_schema_row,
)


def _make_jsonl(rows: list[dict]) -> bytes:
    return "\n".join(json.dumps(r) for r in rows).encode(
        "utf-8"
    )


class TestParseSchemaJsonl:
    def test_parses_indexable_rows(self) -> None:
        data = _make_jsonl(
            [
                {
                    "table_name": "t1",
                    "full_text": "테이블명: t1\n설명: foo",
                    "indexable": True,
                    "has_yaml": True,
                },
                {
                    "table_name": "t2",
                    "full_text": "테이블명: t2\n설명: bar",
                    "indexable": True,
                    "has_yaml": False,
                },
            ]
        )
        rows = parse_schema_jsonl(data)
        assert len(rows) == 2
        assert rows[0].table_name == "t1"
        assert rows[0].has_yaml is True
        assert rows[1].has_yaml is False

    def test_skips_non_indexable(self) -> None:
        data = _make_jsonl(
            [
                {
                    "table_name": "t1",
                    "full_text": "x",
                    "indexable": False,
                },
                {
                    "table_name": "t2",
                    "full_text": "y",
                    "indexable": True,
                },
            ]
        )
        rows = parse_schema_jsonl(data)
        assert [r.table_name for r in rows] == ["t2"]

    def test_skips_blank_lines(self) -> None:
        data = (
            b'{"table_name":"t1","full_text":"x","indexable":true}\n'
            b"\n"
            b'{"table_name":"t2","full_text":"y","indexable":true}\n'
        )
        assert len(parse_schema_jsonl(data)) == 2

    def test_raises_on_bad_json(self) -> None:
        with pytest.raises(ValueError):
            parse_schema_jsonl(b"{not json}")

    def test_raises_on_missing_fields(self) -> None:
        data = _make_jsonl(
            [{"table_name": "t1", "indexable": True}]
        )
        with pytest.raises(ValueError):
            parse_schema_jsonl(data)


class TestSplitSchemaRow:
    def test_short_row_returns_single_chunk(self) -> None:
        text = "테이블명: t1\n설명: short"
        assert split_schema_row(text) == [text]

    def test_long_row_is_split_with_header(self) -> None:
        header = (
            "테이블명: big\n설명: 길고 긴 설명\n키워드: a,b,c"
        )
        columns = "\n".join(
            f"  - col{i} (bigint) [NOT NULL] | description {i} " * 5
            for i in range(30)
        )
        full_text = f"{header}\n컬럼:\n{columns}"
        chunks = split_schema_row(full_text, max_chars=500)
        assert len(chunks) > 1
        for c in chunks:
            assert c.startswith("테이블명: big")
            assert "설명:" in c

    def test_split_preserves_section_label(self) -> None:
        header = "테이블명: t1\n설명: s"
        columns = "\n".join(
            f"  - col{i} (bigint) | very long column description text here {i}"
            for i in range(20)
        )
        relations = "\n".join(
            f"  - col{i} → other.col{i}" for i in range(20)
        )
        full_text = (
            f"{header}\n컬럼:\n{columns}\n관계:\n{relations}"
        )
        chunks = split_schema_row(full_text, max_chars=400)
        assert len(chunks) > 1
        joined = "\n".join(chunks)
        assert "컬럼:" in joined
        assert "관계:" in joined
