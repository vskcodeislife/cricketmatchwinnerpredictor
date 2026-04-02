"""Prediction tracker and self-learning service.

Responsibilities
----------------
1. For every upcoming IPL 2026 match, make a prediction and persist it.
2. After matches complete, look up actual results from cricsheet recently-played
   data and record whether our prediction was correct.
3. When wrong predictions accumulate beyond a threshold, trigger a cricsheet
   download + retrain + model hot-reload (self-learning).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from cricket_predictor.api.schemas import MatchPredictionRequest
from cricket_predictor.config.settings import Settings, get_settings
from cricket_predictor.data.predictions_db import PredictionsDB
from cricket_predictor.providers.cricinfo_standings import venue_advantage
from cricket_predictor.providers.ipl_schedule import IPLScheduleProvider
from cricket_predictor.services.override_parser import apply_overrides, parse_override
from cricket_predictor.services.standings_service import get_standings_service

# Only trust NRR-derived strengths once a team has played this many games;
# smaller samples produce wild swings that mislead the model.
_MIN_GAMES_FOR_NRR = 3

log = logging.getLogger(__name__)

# Retrain when this many consecutive wrong predictions since last retrain
RETRAIN_WRONG_THRESHOLD = 5
# …but only if we have at least this many total predictions logged
RETRAIN_MIN_PREDICTIONS = 8


class PredictionTrackerService:
    """Orchestrates upcoming-match predictions and self-learning retrains."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        db_path = Path(settings.model_artifact_dir).parent.parent / "data" / "predictions.db"
        self._db = PredictionsDB(db_path)
        self._schedule = IPLScheduleProvider()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_upcoming_matches(self) -> list[dict]:
        """Make predictions for matches that don't have one yet.

        Returns a list of newly-created prediction records.
        """
        from cricket_predictor.services.prediction_service import get_prediction_service

        svc = get_prediction_service()
        standings = get_standings_service()
        new_predictions: list[dict] = []

        for match in self._schedule.upcoming_matches():
            if self._db.get_prediction(match["match_id"]):
                continue  # Already predicted

            team_a = match["team_a"]
            team_b = match["team_b"]

            # Pull live standings data for team context
            ta_standing = standings.get_team(team_a)
            tb_standing = standings.get_team(team_b)

            # NRR is meaningless until a team has played a few games; fall back
            # to neutral (65.0) to avoid early-season noise corrupting the model.
            ta_games = ta_standing.played if ta_standing else 0
            tb_games = tb_standing.played if tb_standing else 0
            team_a_form = standings.recent_form(team_a)
            team_b_form = standings.recent_form(team_b)
            team_a_bat  = standings.batting_strength(team_a)  if ta_games >= _MIN_GAMES_FOR_NRR else 65.0
            team_b_bat  = standings.batting_strength(team_b)  if tb_games >= _MIN_GAMES_FOR_NRR else 65.0
            team_a_bowl = standings.bowling_strength(team_a)  if ta_games >= _MIN_GAMES_FOR_NRR else 65.0
            team_b_bowl = standings.bowling_strength(team_b)  if tb_games >= _MIN_GAMES_FOR_NRR else 65.0

            # Apply any active overrides (injuries, pitch notes, etc.)
            active_overrides = self._db.get_active_overrides()
            parsed_overrides = [o["parsed"] for o in active_overrides]
            team_a_bat, team_b_bat, team_a_bowl, team_b_bowl = apply_overrides(
                team_a, team_b, team_a_bat, team_b_bat, team_a_bowl, team_b_bowl,
                parsed_overrides,
            )

            # Home ground advantage (+1.0 = team_a home, -1.0 = team_b home)
            venue_adv = venue_advantage(match["venue"], team_a, team_b)

            request = MatchPredictionRequest(
                team_a=team_a,
                team_b=team_b,
                venue=match["venue"],
                match_format="T20",
                pitch_type="balanced",
                toss_winner="Unknown",  # Toss not known pre-match
                toss_decision="bat",
                team_a_recent_form=team_a_form,
                team_b_recent_form=team_b_form,
                team_a_batting_strength=team_a_bat,
                team_b_batting_strength=team_b_bat,
                team_a_bowling_strength=team_a_bowl,
                team_b_bowling_strength=team_b_bowl,
                head_to_head_win_pct_team_a=0.5,
                venue_advantage_team_a=venue_adv,
                night_match=True,
            )

            result = svc.predict_match(request)
            probs = result["winning_probability"]

            self._db.save_prediction(
                match_id=match["match_id"],
                team_a=team_a,
                team_b=team_b,
                venue=match["venue"],
                match_date=match["match_date"],
                predicted_winner=result["predicted_winner"],
                team_a_probability=probs.get(team_a, 0.5) / 100,
                team_b_probability=probs.get(team_b, 0.5) / 100,
                confidence_score=result["confidence_score"],
                explanation=result.get("explanation", ""),
            )
            log.info(
                "Saved prediction for %s: %s vs %s → %s (%.0f%%)",
                match["match_id"], team_a, team_b,
                result["predicted_winner"],
                max(probs.values()),
            )
            new_predictions.append({
                "match_id": match["match_id"],
                "match": f"{team_a} vs {team_b}",
                "date": match["match_date"],
                "predicted_winner": result["predicted_winner"],
            })

        return new_predictions

    def check_results_and_learn(self) -> dict:
        """Check cricsheet for match results and trigger retraining if needed.

        Returns a summary dict with keys: checked, updated, retrained.
        """
        checked, updated = 0, 0

        pending_ids = self._db.get_pending_result_match_ids()
        if not pending_ids:
            return {"checked": 0, "updated": 0, "retrained": False}

        # Build a lookup of match_id → winner from cricsheet recently-played data
        result_lookup = self._fetch_cricsheet_results()

        for match_id in pending_ids:
            checked += 1
            match = self._schedule.get_match_by_id(match_id)
            if match is None:
                continue
            winner = result_lookup.get(
                (match["team_a"], match["team_b"], match["match_date"])
            ) or result_lookup.get(
                (match["team_b"], match["team_a"], match["match_date"])
            )
            if winner:
                was_correct = self._db.record_result(match_id, winner)
                if was_correct is not None:
                    updated += 1
                    log.info(
                        "Result recorded for %s: actual=%s (%s)",
                        match_id, winner, "✓" if was_correct else "✗",
                    )

        retrained = False
        stats = self._db.get_accuracy_stats()
        if (
            stats["wrong_since_retrain"] >= RETRAIN_WRONG_THRESHOLD
            and stats["total"] >= RETRAIN_MIN_PREDICTIONS
        ):
            log.info(
                "Triggering self-learning retrain (%d wrong since last retrain).",
                stats["wrong_since_retrain"],
            )
            retrained = self._do_retrain()

        return {"checked": checked, "updated": updated, "retrained": retrained}

    def get_next_match_prediction(self) -> dict | None:
        """Return the full prediction record for the next scheduled match."""
        nxt = self._schedule.next_match()
        if nxt is None:
            return None
        saved = self._db.get_prediction(nxt["match_id"])
        if saved:
            return {**nxt, **saved}
        return dict(nxt)

    def get_recent_history(self, limit: int = 10) -> list[dict]:
        return self._db.get_recent_predictions(limit)

    def get_accuracy_stats(self) -> dict:
        return self._db.get_accuracy_stats()

    # ------------------------------------------------------------------
    # Override management (injuries, pitch notes, etc.)
    # ------------------------------------------------------------------

    def add_override(self, note: str) -> list[dict]:
        """Parse a free-text note and persist it.  Returns parsed adjustments."""
        from cricket_predictor.services.override_parser import parse_override
        parsed = parse_override(note)
        if parsed:
            for adj in parsed:
                self._db.save_override(note=note, parsed=adj)
        else:
            # Store as an unrecognised note (no feature adjustments) so it shows
            # in the UI for the user to see.
            self._db.save_override(
                note=note,
                parsed={"type": "note", "team": None, "player": None,
                        "role": None, "bowl_delta": 0.0, "bat_delta": 0.0,
                        "description": f"📝 Note: {note}"},
            )
        # Delete future predictions so they are re-generated with the new overrides
        self._invalidate_future_predictions()
        return parsed

    def get_active_overrides(self) -> list[dict]:
        return self._db.get_active_overrides()

    def delete_override(self, override_id: int) -> None:
        self._db.delete_override(override_id)
        self._invalidate_future_predictions()

    def refresh_injury_overrides(self) -> int:
        """Fetch injury report from crictracker and sync overrides.

        Clears any existing injury-sourced overrides, fetches the latest
        report, parses each entry, and saves as new overrides.  Returns the
        number of injury entries found.
        """
        from cricket_predictor.providers.injury_report_provider import InjuryReportProvider

        provider = InjuryReportProvider()
        try:
            report = provider.fetch()
            provider.save(report)
        except Exception as exc:
            log.warning("Injury report fetch failed: %s", exc)
            report = provider.load()
            if not report:
                return 0

        # Clear old auto-generated injury overrides
        existing = self._db.get_active_overrides()
        for ov in existing:
            if ov.get("note", "").startswith("[auto-injury]"):
                self._db.delete_override(ov["id"])

        # Parse the injury report into override text and apply
        override_text = provider.build_override_text(report)
        if not override_text:
            return 0

        parsed = parse_override(override_text)
        for adj in parsed:
            self._db.save_override(note="[auto-injury] " + adj.get("description", ""), parsed=adj)

        # Re-generate predictions with new overrides
        if parsed:
            self._invalidate_future_predictions()
            log.info("Injury overrides refreshed: %d adjustments from %d players",
                     len(parsed), report.get("total_unavailable", 0))

        return len(parsed)

    def _invalidate_future_predictions(self) -> None:
        """Delete all future predictions so they are re-generated with latest overrides."""
        with self._db._connect() as conn:
            conn.execute("DELETE FROM match_predictions WHERE match_date >= date('now')")

    def _fetch_cricsheet_results(self) -> dict[tuple[str, str, str], str]:
        """Parse recently-played cricsheet JSON files to extract match winners.

        Returns a dict keyed by (team_a, team_b, match_date) → winning team name.
        """
        from cricket_predictor.providers.ipl_schedule import SHORT_TEAM

        results: dict[tuple[str, str, str], str] = {}
        recently_played_dir = Path(self._settings.cricsheet_data_dir) / "recently_played_30_male_json"
        if not recently_played_dir.exists():
            return results

        import json

        for match_file in recently_played_dir.glob("*.json"):
            try:
                data = json.loads(match_file.read_text())
                info = data.get("info", {})
                match_type = info.get("match_type", "")
                teams = info.get("teams", [])
                if match_type != "T20" or len(teams) != 2:
                    continue
                dates = info.get("dates", [])
                if not dates:
                    continue
                outcome = info.get("outcome", {})
                winner_raw = outcome.get("winner")
                if not winner_raw:
                    continue
                # Normalise team name to full name via SHORT_TEAM aliases
                winner = SHORT_TEAM.get(winner_raw, winner_raw)
                team_a_raw, team_b_raw = teams
                team_a = SHORT_TEAM.get(team_a_raw, team_a_raw)
                team_b = SHORT_TEAM.get(team_b_raw, team_b_raw)
                results[(team_a, team_b, dates[0])] = winner
                results[(team_b, team_a, dates[0])] = winner
            except Exception:  # noqa: BLE001
                continue
        return results

    def _do_retrain(self) -> bool:
        """Download fresh cricsheet data, retrain, and hot-reload models."""
        try:
            from cricket_predictor.services.data_update_service import DataUpdateService
            from cricket_predictor.services.prediction_service import get_prediction_service

            svc = DataUpdateService(self._settings)
            retrained = svc.check_and_retrain()
            if retrained:
                get_prediction_service().reload_models()
                self._db.mark_retrained()
                log.info("Self-learning retrain completed and models hot-reloaded.")
            return retrained
        except Exception as exc:  # noqa: BLE001
            log.error("Self-learning retrain failed: %s", exc)
            return False


@lru_cache
def get_prediction_tracker(settings: Settings | None = None) -> PredictionTrackerService:
    return PredictionTrackerService(settings or get_settings())
