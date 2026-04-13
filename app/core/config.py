from functools import cached_property

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
    opensearch_schema_index: str = "kasanova_schema"
    llm_provider: str = "ollama"
    llm_server_url: str = "http://localhost:11434"
    llm_model_name: str = "phi3"
    llm_api_key: str = ""
    TAVILY_API_KEY: str
    redis_url: str = "redis://localhost:6379/0"
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_db: str = ""
    mysql_user: str = ""
    mysql_password: str = ""

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
