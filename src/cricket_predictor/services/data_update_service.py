"""Orchestrates the daily cricsheet update check and model retrain cycle.

Callers should use ``check_and_retrain()`` which:
  1. Issues cheap HTTP HEAD requests to each configured cricsheet URL.
  2. Downloads and extracts the ZIP only when the Content-Length changed.
  3. Parses the match and player records from the extracted JSON files.
  4. Retrains the pipeline and saves new model artifacts.

Returns ``True`` when artifacts were actually updated so callers can decide
whether to hot-reload the running prediction service.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cricket_predictor.config.settings import Settings
from cricket_predictor.data.cricsheet_loader import CricsheetLoader
from cricket_predictor.models.training import save_artifacts, train_all

log = logging.getLogger(__name__)


class DataUpdateService:
    """Check cricsheet archives for updates, retrain models if data is fresh."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._loader = CricsheetLoader(settings.cricsheet_data_dir)

    def check_and_retrain(self) -> bool:
        """Return True when models are successfully retrained from fresh data."""
        urls: list[str] = [
            u
            for u in [
                self._settings.cricsheet_ipl_url,
                self._settings.cricsheet_t20_url,
                self._settings.cricsheet_recent_url,
            ]
            if u
        ]
        if not urls:
            log.info("No cricsheet URLs configured – skipping update check.")
            return False

        if not self._loader.check_for_updates(urls):
            log.info("No cricsheet updates detected.")
            return False

        log.info("New data detected – downloading and retraining …")
        extracted = self._loader.download_and_extract(urls)
        if not extracted:
            log.warning("No directories extracted – skipping retrain.")
            return False

        matches_df = self._loader.parse_matches(extracted)
        players_df = self._loader.parse_player_stats(extracted)

        if matches_df.empty or players_df.empty:
            log.warning(
                "Parsed data is empty (matches=%d, players=%d) – skipping retrain.",
                len(matches_df),
                len(players_df),
            )
            return False

        log.info("Training on %d matches and %d players.", len(matches_df), len(players_df))
        artifacts = train_all(matches_df, players_df)
        save_artifacts(artifacts, self._settings.model_artifact_dir)
        log.info("Models retrained and saved to %s.", self._settings.model_artifact_dir)
        return True

    def get_status(self) -> dict[str, Any]:
        """Return current metadata about each tracked cricsheet source."""
        return {
            "tracked_sources": self._loader.get_meta(),
            "cricsheet_data_dir": str(Path(self._settings.cricsheet_data_dir).resolve()),
        }
