"""Orchestrates data refreshes and retrains for the match and player models.

``check_and_retrain()`` handles the remote cricsheet update path, while
``retrain_from_local_data()`` reuses the locally extracted JSON archives and
augments match training with completed prediction-history examples.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from cricket_predictor.config.settings import Settings
from cricket_predictor.data.cricsheet_loader import CricsheetLoader
from cricket_predictor.data.predictions_db import PredictionsDB, default_predictions_db_path
from cricket_predictor.models.training import (
    load_match_model,
    save_artifacts,
    score_match_model,
    train_all,
    train_match_model,
)
from cricket_predictor.providers.cricinfo_standings import resolve_team_name
from cricket_predictor.providers.ipl_csv_provider import IplCsvDataProvider, TeamLeaderStats

log = logging.getLogger(__name__)


class DataUpdateService:
    """Retrain models from cricsheet data and completed prediction feedback."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._loader = CricsheetLoader(settings.cricsheet_data_dir)
        self._predictions_db = PredictionsDB(default_predictions_db_path(settings.model_artifact_dir))

    def check_and_retrain(self) -> bool:
        """Return True when models are successfully retrained from fresh data."""
        urls = self._configured_urls()
        if not urls:
            log.info("No cricsheet URLs configured – skipping update check.")
            return False

        if not self._loader.check_for_updates(urls):
            log.info("No cricsheet updates detected.")
            return False

        log.info("New data detected – downloading and retraining …")
        extracted = self._loader.download_and_extract(urls)
        return self._retrain_from_dirs(extracted, source_label="fresh cricsheet data")

    def retrain_from_cricsheet(self, download: bool = False) -> bool:
        """Retrain from cricsheet archives, optionally downloading fresh copies first."""
        urls = self._configured_urls()
        if not urls:
            log.warning("No cricsheet URLs configured – skipping retrain.")
            return False

        if download:
            extracted = self._loader.download_and_extract(urls)
            source_label = "downloaded cricsheet data"
        else:
            extracted = self._get_local_json_dirs(urls)
            source_label = "local cricsheet data"

        if not extracted:
            log.warning("No local cricsheet data found – skipping retrain.")
            return False
        return self._retrain_from_dirs(extracted, source_label=source_label)

    def retrain_from_local_data(self) -> bool:
        """Retrain using locally extracted cricsheet data plus completed predictions."""
        return self.retrain_from_cricsheet(download=False)

    def get_status(self) -> dict[str, Any]:
        """Return current metadata about each tracked cricsheet source."""
        return {
            "tracked_sources": self._loader.get_meta(),
            "cricsheet_data_dir": str(Path(self._settings.cricsheet_data_dir).resolve()),
        }

    def _configured_urls(self) -> list[str]:
        return [
            url
            for url in [
                self._settings.cricsheet_ipl_url,
                self._settings.cricsheet_t20_url,
                self._settings.cricsheet_recent_url,
            ]
            if url
        ]

    def _get_local_json_dirs(self, urls: list[str]) -> list[Path]:
        root = Path(self._settings.cricsheet_data_dir)
        meta = self._loader.get_meta()
        candidates: list[Path] = []

        for url in urls:
            source_dir = meta.get(url, {}).get("source_dir")
            if source_dir:
                candidates.append(Path(source_dir))
            candidates.append(root / Path(url).stem)

        if root.exists():
            candidates.extend(path for path in root.iterdir() if path.is_dir())

        unique_dirs: list[Path] = []
        seen: set[Path] = set()
        for directory in candidates:
            resolved = directory.resolve()
            if resolved in seen or not directory.exists() or not directory.is_dir():
                continue
            seen.add(resolved)
            unique_dirs.append(directory)
        return unique_dirs

    def _augment_matches_with_feedback(self, matches_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        feedback_rows = self._predictions_db.get_feedback_training_rows()
        if not feedback_rows:
            return matches_df, 0

        feedback_df = pd.DataFrame(feedback_rows)
        feedback_df = self._backfill_feedback_leader_signals(feedback_df).assign(_feedback_example=1)
        base_df = matches_df.copy().assign(_feedback_example=0)
        augmented = pd.concat([base_df, feedback_df], ignore_index=True, sort=False)
        key_columns = {"team_a", "team_b", "match_date"}
        if key_columns.issubset(augmented.columns):
            augmented = (
                augmented.sort_values(["match_date", "_feedback_example"])
                .drop_duplicates(subset=["team_a", "team_b", "match_date"], keep="last")
                .reset_index(drop=True)
            )
        augmented = augmented.drop(columns=["_feedback_example"], errors="ignore")
        return augmented, len(feedback_df)

    def _backfill_feedback_leader_signals(self, feedback_df: pd.DataFrame) -> pd.DataFrame:
        if feedback_df.empty:
            return feedback_df

        # Prefer live iplt20 S3 feeds; fall back to Kaggle CSV
        from cricket_predictor.providers.iplt20_stats_provider import fetch_team_leader_stats

        leader_lookup = fetch_team_leader_stats(self._settings.iplt20_stats_competition_id)
        if not leader_lookup and self._settings.ipl_csv_data_dir:
            leader_lookup = IplCsvDataProvider(self._settings.ipl_csv_data_dir).team_leader_stats_lookup()
        if not leader_lookup:
            return feedback_df

        enriched = feedback_df.copy()
        for column in (
            "team_a_top_run_getters_runs",
            "team_b_top_run_getters_runs",
            "team_a_top_wicket_takers_wickets",
            "team_b_top_wicket_takers_wickets",
        ):
            if column not in enriched.columns:
                enriched[column] = pd.NA

        self._fill_leader_column(
            enriched,
            team_column="team_a",
            value_column="team_a_top_run_getters_runs",
            extractor=lambda stats: stats.top_run_getters_runs,
            leader_lookup=leader_lookup,
        )
        self._fill_leader_column(
            enriched,
            team_column="team_b",
            value_column="team_b_top_run_getters_runs",
            extractor=lambda stats: stats.top_run_getters_runs,
            leader_lookup=leader_lookup,
        )
        self._fill_leader_column(
            enriched,
            team_column="team_a",
            value_column="team_a_top_wicket_takers_wickets",
            extractor=lambda stats: stats.top_wicket_takers_wickets,
            leader_lookup=leader_lookup,
        )
        self._fill_leader_column(
            enriched,
            team_column="team_b",
            value_column="team_b_top_wicket_takers_wickets",
            extractor=lambda stats: stats.top_wicket_takers_wickets,
            leader_lookup=leader_lookup,
        )
        return enriched

    def _fill_leader_column(
        self,
        frame: pd.DataFrame,
        *,
        team_column: str,
        value_column: str,
        extractor,
        leader_lookup: dict[str, TeamLeaderStats],
    ) -> None:
        numeric = pd.to_numeric(frame[value_column], errors="coerce")
        needs_fill = numeric.isna() | (numeric <= 0)
        if not needs_fill.any():
            return

        fallback_values = frame[team_column].map(
            lambda team: extractor(leader_lookup.get(resolve_team_name(str(team)), TeamLeaderStats()))
        )
        frame.loc[needs_fill, value_column] = fallback_values[needs_fill]

    # -- Validation gate --------------------------------------------------

    _VALIDATION_TOLERANCE = 0.05  # Accept if new accuracy >= old - 5%
    _MIN_VALIDATION_SAMPLES = 15

    def _validate_retrain(self, matches_df: pd.DataFrame) -> bool:
        """Return True if a model trained on this data passes quality checks.

        Uses a chronological train/validation split to compare the candidate
        model against the currently deployed model.  Skips validation when the
        dataset is too small for a meaningful held-out set.
        """
        min_val = self._MIN_VALIDATION_SAMPLES
        if len(matches_df) < min_val * 2:
            log.info("Dataset too small (%d rows) for validation \u2014 accepting retrain.", len(matches_df))
            return True

        val_size = max(int(len(matches_df) * 0.2), min_val)
        sorted_df = matches_df.sort_values("match_date") if "match_date" in matches_df.columns else matches_df
        val_df = sorted_df.tail(val_size)
        train_df = sorted_df.head(len(sorted_df) - val_size)

        try:
            candidate = train_match_model(train_df)
            new_score = score_match_model(candidate, val_df)
        except Exception as exc:
            log.warning("Validation training failed (%s) \u2014 accepting retrain.", exc)
            return True

        old_score = self._score_existing_model(val_df)
        if old_score is None:
            log.info("No existing model to compare \u2014 accepting retrain (new=%.1f%%).", new_score * 100)
            return True

        if new_score < old_score - self._VALIDATION_TOLERANCE:
            log.warning(
                "Retrain rejected: new model (%.1f%%) worse than current (%.1f%%) on %d held-out matches.",
                new_score * 100, old_score * 100, len(val_df),
            )
            return False

        log.info(
            "Retrain validated: new=%.1f%% vs current=%.1f%% on %d held-out matches.",
            new_score * 100, old_score * 100, len(val_df),
        )
        return True

    def _score_existing_model(self, val_df: pd.DataFrame) -> float | None:
        """Score the currently deployed match model on validation data."""
        model_path = Path(self._settings.model_artifact_dir) / "match_model.joblib"
        if not model_path.exists():
            return None
        try:
            existing = load_match_model(self._settings.model_artifact_dir)
            return score_match_model(existing, val_df)
        except Exception:
            return None

    def _backup_artifacts(self) -> None:
        """Copy current model files to *.prev before overwriting."""
        model_dir = Path(self._settings.model_artifact_dir)
        for name in ("match_model.joblib", "player_model.joblib"):
            src = model_dir / name
            if src.exists():
                try:
                    shutil.copy2(src, model_dir / f"{name}.prev")
                except Exception as exc:
                    log.warning("Failed to backup %s: %s", src, exc)

    def _retrain_from_dirs(self, json_dirs: list[Path], source_label: str) -> bool:
        if not json_dirs:
            log.warning("No directories available for retrain from %s.", source_label)
            return False

        matches_df = self._loader.parse_matches(json_dirs)
        players_df = self._loader.parse_player_stats(json_dirs)
        matches_df, feedback_count = self._augment_matches_with_feedback(matches_df)

        if matches_df.empty or players_df.empty:
            log.warning(
                "Parsed data is empty after feedback merge (matches=%d, players=%d) – skipping retrain.",
                len(matches_df),
                len(players_df),
            )
            return False

        if not self._validate_retrain(matches_df):
            return False

        log.info(
            "Training on %d matches (%d completed prediction examples) and %d players from %s.",
            len(matches_df),
            feedback_count,
            len(players_df),
            source_label,
        )
        self._backup_artifacts()
        artifacts = train_all(matches_df, players_df)
        save_artifacts(artifacts, self._settings.model_artifact_dir)
        log.info("Models retrained and saved to %s.", self._settings.model_artifact_dir)
        return True
