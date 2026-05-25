"""Application configuration via environment + .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime config. Read once at startup; never mutate."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    alpaca_api_key: str = Field(..., description="Alpaca paper API key")
    # Accept any of the three common Alpaca env naming conventions:
    #   ALPACA_SECRET_KEY    (alpaca-py SDK convention — preferred)
    #   ALPACA_API_SECRET    (older Alpaca docs)
    #   ALPACA_API_SECRET_KEY (defensive — both prefixes)
    alpaca_secret_key: str = Field(
        ...,
        description="Alpaca paper secret key",
        validation_alias=AliasChoices(
            "alpaca_secret_key",
            "alpaca_api_secret",
            "alpaca_api_secret_key",
        ),
    )
    alpaca_paper: bool = Field(default=True, description="Use paper account")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca REST base URL",
    )

    fred_api_key: str = Field(..., description="FRED API key")

    log_level: str = Field(default="INFO", description="loguru level")
    data_dir: Path = Field(
        default=Path("./data"),
        description="Root data directory",
        validation_alias="QUANT_DATA_DIR",
    )
