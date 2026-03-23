from unittest.mock import patch, MagicMock

import pytest
from langchain_core.language_models import BaseChatModel


class TestGetLlm:
    def test_ollama_returns_chat_ollama(self) -> None:
        mock_settings = MagicMock(
            llm_provider="ollama",
            llm_server_url="http://localhost:11434",
            llm_model_name="phi3",
        )
        with patch("app.core.llm.get_settings", return_value=mock_settings):
            from app.core.llm import get_llm

            llm = get_llm()

        from langchain_ollama import ChatOllama

        assert isinstance(llm, ChatOllama)
        assert isinstance(llm, BaseChatModel)

    def test_openai_returns_chat_openai(self) -> None:
        mock_settings = MagicMock(
            llm_provider="openai",
            llm_model_name="gpt-4o",
            llm_api_key="test-key",
        )
        with patch("app.core.llm.get_settings", return_value=mock_settings):
            from app.core.llm import get_llm

            llm = get_llm()

        from langchain_openai import ChatOpenAI

        assert isinstance(llm, ChatOpenAI)
        assert isinstance(llm, BaseChatModel)

    def test_anthropic_returns_chat_anthropic(self) -> None:
        mock_settings = MagicMock(
            llm_provider="anthropic",
            llm_model_name="claude-sonnet-4-20250514",
            llm_api_key="test-key",
        )
        with patch("app.core.llm.get_settings", return_value=mock_settings):
            from app.core.llm import get_llm

            llm = get_llm()

        from langchain_anthropic import ChatAnthropic

        assert isinstance(llm, ChatAnthropic)
        assert isinstance(llm, BaseChatModel)

    def test_unsupported_provider_raises_value_error(self) -> None:
        mock_settings = MagicMock(llm_provider="unknown")
        with patch("app.core.llm.get_settings", return_value=mock_settings):
            from app.core.llm import get_llm

            with pytest.raises(ValueError, match="지원하지 않는"):
                get_llm()
