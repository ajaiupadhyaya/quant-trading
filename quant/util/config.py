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

    # Off-box alerting (E1). Optional so CI's dummy env still constructs Settings.
    # Field names use the singular "healthcheck_" prefix (per the E1 design's shared
    # types) but the env vars use the plural "HEALTHCHECKS_" prefix, so alias them.
    healthcheck_tick_url: str | None = Field(
        default=None,
        description="healthchecks.io tick ping URL",
        validation_alias=AliasChoices("healthcheck_tick_url", "healthchecks_tick_url"),
    )
    healthcheck_guard_url: str | None = Field(
        default=None,
        description="healthchecks.io guard ping URL",
        validation_alias=AliasChoices("healthcheck_guard_url", "healthchecks_guard_url"),
    )
    pushover_app_token: str | None = Field(default=None, description="Pushover application token")
    pushover_user_key: str | None = Field(default=None, description="Pushover user key")

    # Slack delivery for the daily analyst digest + real-time alerts (E2). Optional.
    slack_webhook_url: str | None = Field(
        default=None, description="Slack Incoming Webhook URL for digests + alerts"
    )
    # Claude API for the analyst digest narration (E2). Optional so the digest
    # degrades to a deterministic, template-only summary when unset.
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key (analyst)")
    anthropic_model: str = Field(
        default="claude-opus-4-8", description="Claude model for high-stakes judgment (daily brief)"
    )
    anthropic_model_fast: str = Field(
        default="claude-haiku-4-5",
        description="Cheaper Claude model for routine/high-frequency calls (intraday watch, "
        "daily digest) — keeps cost low; Opus is reserved for the brief/weekly synthesis",
    )

    # Portfolio risk gate (Guard 5) mode. 'warn' (default) records would-be
    # VaR/CVaR/vol/beta/asset-class violations without ever blocking the live
    # batch; 'block' refuses the batch on violation (a deliberate, human-gated
    # flip — never enabled without a clean WARN bake-in). A bad value fails open.
    portfolio_risk_gate_mode: str = Field(
        default="warn",
        description="Portfolio risk Guard 5 mode: 'warn' (record only) or 'block'",
        validation_alias="QUANT_PORTFOLIO_RISK_GATE_MODE",
    )

    log_level: str = Field(default="INFO", description="loguru level")
    data_dir: Path = Field(
        default=Path("./data"),
        description="Root data directory",
        validation_alias="QUANT_DATA_DIR",
    )
