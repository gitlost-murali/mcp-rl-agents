from typing import Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mcp_url: str = Field(default="http://localhost:1234")
    mcp_bearer_token: str = Field(default="")
    openrouter_key: str = Field(default="")
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    wandb_api_key: str | None = None

    @classmethod
    def load(cls, **kwargs: Any) -> "Settings":  # noqa: D401 – clear enough
        """Return a fully validated Settings instance or exit with details.

        Encapsulates the try/except that was previously at module scope so we
        can construct the settings object with one concise call while still
        providing the same UX when environment variables are missing.
        """
        try:
            return cls(**kwargs)
        except Exception as exc:
            raise ValueError(exc)


settings = Settings.load()
