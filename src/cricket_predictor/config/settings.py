from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Cricket Predictor API"
    app_version: str = "0.1.0"
    model_artifact_dir: str = "artifacts/models"
    synthetic_data_dir: str = "data/synthetic"

    # --- Cricsheet automatic data refresh ---
    cricsheet_data_dir: str = "data/cricsheet"
    cricsheet_ipl_url: str = "https://cricsheet.org/downloads/ipl_male_json.zip"
    cricsheet_t20_url: str = "https://cricsheet.org/downloads/t20s_male_json.zip"
    cricsheet_recent_url: str = "https://cricsheet.org/downloads/recently_played_30_male_json.zip"
    enable_cricsheet_updates: bool = False
    cricsheet_check_interval_hours: int = Field(default=24, ge=1)

    # --- IPL standings (points table) ---
    # Source: Delhi Capitals page — embeds live Cricbuzz SSR JSON, no JS needed.
    cricinfo_standings_url: str = "https://www.delhicapitals.in/points-table"
    enable_standings_refresh: bool = True
    standings_refresh_minutes: int = Field(default=30, ge=5)

    # --- In-process live predictions ---
    live_provider: str = "mock"
    live_refresh_seconds: int = Field(default=60, ge=15)
    live_provider_base_url: str | None = None
    enable_live_updates: bool = False

    # --- Prediction tracker (upcoming matches + self-learning) ---
    tracker_interval_seconds: int = Field(default=3600, ge=60, description="How often to check results and retrain (seconds)")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CRICKET_PREDICTOR_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
