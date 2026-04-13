import json
from dataclasses import dataclass


@dataclass
class SchemaRow:
    table_name: str
    full_text: str
    has_yaml: bool


def parse_schema_jsonl(content: bytes) -> list[SchemaRow]:
    """JSONL 바이트를 파싱해 indexable한 행만 반환한다."""
    rows: list[SchemaRow] = []
    for raw in content.decode("utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSONL 라인 파싱 실패: {e}"
            ) from e
        if not obj.get("indexable", False):
            continue
        table_name = obj.get("table_name")
        full_text = obj.get("full_text")
        if not table_name or not full_text:
            raise ValueError(
                "table_name/full_text 누락 행 존재"
            )
        rows.append(
            SchemaRow(
                table_name=table_name,
                full_text=full_text,
                has_yaml=bool(obj.get("has_yaml", False)),
            )
        )
    return rows


def split_schema_row(
    full_text: str, max_chars: int = 1536
) -> list[str]:
    """헤더(테이블명/설명/키워드)를 보존하며 행을 청크로 분할한다.

    max_chars 이하면 원문 그대로 1개 청크로 반환한다.
    초과하면 `컬럼:`/`관계:` 섹션 경계에서 불릿 단위로 잘라
    각 sub-chunk 앞에 헤더를 prepend한다.
    """
    if len(full_text) <= max_chars:
        return [full_text]

    lines = full_text.split("\n")
    body_start = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith("컬럼:") or line.startswith("관계:")
        ),
        len(lines),
    )
    header = "\n".join(lines[:body_start]).rstrip()
    body_lines = lines[body_start:]
    if not header or not body_lines:
        return [full_text]

    chunks: list[str] = []
    current_lines: list[str] = [header]
    current_section: str | None = None

    def flush() -> None:
        if len(current_lines) > 1:
            chunks.append("\n".join(current_lines))

    for line in body_lines:
        is_section = line.endswith(":") and not line.startswith(
            " "
        )
        if is_section:
            current_section = line
            candidate = "\n".join(current_lines + [line])
            if (
                len(candidate) > max_chars
                and len(current_lines) > 1
            ):
                flush()
                current_lines = [header, line]
            else:
                current_lines.append(line)
            continue

        candidate = "\n".join(current_lines + [line])
        if (
            len(candidate) > max_chars
            and len(current_lines) > 1
        ):
            flush()
            current_lines = [header]
            if current_section:
                current_lines.append(current_section)
            current_lines.append(line)
        else:
            current_lines.append(line)

    flush()
    return chunks if chunks else [full_text]
