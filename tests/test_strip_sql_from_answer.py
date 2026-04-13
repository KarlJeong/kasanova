from app.api.slack import _strip_sql_from_answer


def test_strips_sql_code_fence_before_answer():
    raw = (
        "```sql\n"
        "SELECT\n"
        "  COUNT(DISTINCT id)\n"
        "FROM kasa_auth_user\n"
        "WHERE\n"
        "  last_login >= DATE_FORMAT(NOW(), '%Y-%m-01');\n"
        "```\n"
        "이번 달에 접속한 유니크 사용자 수는 0명입니다."
    )
    result = _strip_sql_from_answer(raw)
    assert "SELECT" not in result
    assert "```" not in result
    assert result == "이번 달에 접속한 유니크 사용자 수는 0명입니다."


def test_strips_plain_fence_without_lang_tag():
    raw = (
        "```\nSELECT 1 FROM dual;\n```\n"
        "결과는 1입니다."
    )
    result = _strip_sql_from_answer(raw)
    assert result == "결과는 1입니다."


def test_strips_half_open_fence_and_loose_sql_lines():
    raw = (
        "```sql\n"
        "SELECT name FROM kasa_member\n"
        "WHERE status = 'ACTIVE'\n"
        "활성 회원은 홍길동 외 2명입니다."
    )
    result = _strip_sql_from_answer(raw)
    assert "SELECT" not in result
    assert "WHERE" not in result
    assert "```" not in result
    assert "활성 회원은 홍길동 외 2명입니다." in result


def test_preserves_plain_answer_without_sql():
    raw = "이번 달 신규 가입자는 42명입니다."
    assert _strip_sql_from_answer(raw) == raw


def test_keeps_non_sql_inline_code():
    raw = "설정값은 `max_tokens`이며, 기본값은 4096입니다."
    result = _strip_sql_from_answer(raw)
    assert "max_tokens" in result
    assert "4096" in result
