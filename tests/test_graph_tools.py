from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.graph.tools import create_web_search_tool


class TestWebSearchTool:
    @pytest.fixture
    def tool(self) -> BaseTool:
        return create_web_search_tool()

    def test_is_langchain_base_tool(
        self, tool: BaseTool
    ) -> None:
        assert isinstance(tool, BaseTool)

    @patch("app.graph.tools.TavilyClient")
    @patch("app.graph.tools.get_settings")
    def test_returns_dict(
        self,
        mock_settings: MagicMock,
        mock_tavily_cls: MagicMock,
        tool: BaseTool,
    ) -> None:
        mock_settings.return_value.TAVILY_API_KEY = "tvly-test-key"
        mock_tavily_cls.return_value.search.return_value = {
            "results": [
                {"title": "Test", "url": "https://example.com"}
            ]
        }

        result = tool.invoke({"query": "test query"})

        assert isinstance(result, dict)

    @patch("app.graph.tools.TavilyClient")
    @patch("app.graph.tools.get_settings")
    def test_results_field_exists(
        self,
        mock_settings: MagicMock,
        mock_tavily_cls: MagicMock,
        tool: BaseTool,
    ) -> None:
        mock_settings.return_value.TAVILY_API_KEY = "tvly-test-key"
        mock_tavily_cls.return_value.search.return_value = {
            "results": [
                {"title": "Test", "url": "https://example.com"}
            ]
        }

        result = tool.invoke({"query": "test query"})

        assert "results" in result

    @patch("app.graph.tools.TavilyClient")
    @patch("app.graph.tools.get_settings")
    def test_calls_tavily_client_with_query(
        self,
        mock_settings: MagicMock,
        mock_tavily_cls: MagicMock,
        tool: BaseTool,
    ) -> None:
        mock_settings.return_value.TAVILY_API_KEY = "tvly-test-key"
        mock_client = mock_tavily_cls.return_value

        tool.invoke({"query": "test query"})

        mock_tavily_cls.assert_called_once_with(
            api_key="tvly-test-key"
        )
        mock_client.search.assert_called_once_with("test query")

    def test_missing_tavily_api_key_raises_validation_error(
        self,
    ) -> None:
        from app.core.config import Settings

        with pytest.raises(ValidationError):
            Settings(
                TAVILY_API_KEY=None,  # type: ignore[arg-type]
                database_url="postgresql+asyncpg://x:x@localhost/x",
            )
