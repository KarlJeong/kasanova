from functools import cached_property

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "kasanova-api"
    debug: bool = False
    database_url: str = (
        "postgresql+asyncpg://kasanova:secret@localhost:5433/kasanova"
    )
    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "kasanova_docs"
    opensearch_extra_indices: list[str] = []
    llm_provider: str = "ollama"
    llm_server_url: str = "http://localhost:11434"
    llm_model_name: str = "phi3"
    llm_api_key: str = ""
    TAVILY_API_KEY: str
    redis_url: str = "redis://localhost:6379/0"

    @field_validator("opensearch_extra_indices", mode="before")
    @classmethod
    def _split_indices(cls, v: object) -> object:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @cached_property
    def allowed_indices(self) -> list[str]:
        seen: dict[str, None] = {self.opensearch_index: None}
        for name in self.opensearch_extra_indices:
            seen[name] = None
        return list(seen.keys())

    @cached_property
    def checkpoint_db_url(self) -> str:
        url = self.database_url.replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        return url.replace("ssl=disable", "sslmode=disable")

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()


def get_settings() -> Settings:
    return settings
