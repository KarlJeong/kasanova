from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "kasanova-api"
    debug: bool = False
    database_url: str = "postgresql+asyncpg://kasanova:secret@localhost:5433/kasanova"
    test_database_url: str = "postgresql+asyncpg://kasanova:secret@localhost:5433/kasanova_test"
    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "kasanova-docs"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
