from langchain_core.language_models import BaseChatModel

from app.core.config import get_settings


def get_llm() -> BaseChatModel:
    s = get_settings()
    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=s.llm_server_url, model=s.llm_model_name
        )
    elif s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=s.llm_api_key, model=s.llm_model_name
        )
    elif s.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            api_key=s.llm_api_key, model=s.llm_model_name
        )
    elif s.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=s.llm_api_key,
            model=s.llm_model_name,
        )
    else:
        raise ValueError(
            f"지원하지 않는 LLM_PROVIDER: {s.llm_provider}"
        )
