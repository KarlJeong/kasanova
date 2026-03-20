from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient


async def test_health_ok(client: AsyncClient) -> None:
    """GET /health returns 200 with status ok when DB is reachable."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("app.main.engine") as mock_engine:
        mock_engine.connect.return_value = mock_cm

        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "kasanova-api"


async def test_health_db_failure(client: AsyncClient) -> None:
    """GET /health returns 500 when DB connection fails."""
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(
        side_effect=Exception("connection refused")
    )
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("app.main.engine") as mock_engine:
        mock_engine.connect.return_value = mock_cm

        response = await client.get("/health")

    assert response.status_code == 500
    data = response.json()
    assert data["status"] == "error"
    assert "detail" in data
