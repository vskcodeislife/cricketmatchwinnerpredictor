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
import re
from functools import lru_cache
from pathlib import Path

from cricket_predictor.api.schemas import MatchPredictionRequest
from cricket_predictor.config.settings import Settings, get_settings
from cricket_predictor.data.predictions_db import PredictionsDB, default_predictions_db_path
from cricket_predictor.providers.cricinfo_standings import (
    CricinfoStandingsProvider,
    build_recent_results_lookup,
    resolve_team_name,
    venue_advantage,
)
from cricket_predictor.providers.ipl_csv_provider import IplCsvDataProvider
from cricket_predictor.providers.ipl_schedule import IPLScheduleProvider
from cricket_predictor.services.match_context_service import get_match_context_service
from cricket_predictor.services.override_parser import apply_overrides, parse_override
from cricket_predictor.services.standings_service import get_standings_service

# Only trust NRR-derived strengths once a team has played this many games;
# smaller samples produce wild swings that mislead the model.
_MIN_GAMES_FOR_NRR = 3

log = logging.getLogger(__name__)

_PLAYER_MENTION_PATTERN = re.compile(
    r"\b(?:like|such as|including|featuring|with)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*(?:\s*(?:,|and)\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)*)"
)

# Retrain when this many consecutive wrong predictions since last retrain
RETRAIN_WRONG_THRESHOLD = 5
# …but only if we have at least this many total predictions logged
RETRAIN_MIN_PREDICTIONS = 8
# Also retrain after a few completed matches so correct calls reinforce the model.
RETRAIN_FEEDBACK_THRESHOLD = 3


class PredictionTrackerService:
    """Orchestrates upcoming-match predictions and feedback-driven retrains."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db = PredictionsDB(default_predictions_db_path(settings.model_artifact_dir))
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
        context_service = get_match_context_service(self._settings)
        new_predictions: list[dict] = []

        for match in self._schedule.upcoming_matches():
            if self._db.get_prediction(match["match_id"]):
                continue  # Already predicted

            team_a = match["team_a"]
            team_b = match["team_b"]

            request = context_service.build_request(
                team_a=team_a,
                team_b=team_b,
                venue=match["venue"],
                match_format="T20",
                pitch_type="balanced",
                toss_winner="Unknown",
                toss_decision="bat",
                night_match=True,
            )

            # Apply any active overrides (injuries, pitch notes, etc.)
            active_overrides = self._db.get_active_overrides()
            parsed_overrides = [o["parsed"] for o in active_overrides]
            team_a_bat, team_b_bat, team_a_bowl, team_b_bowl = apply_overrides(
                team_a,
                team_b,
                request.team_a_batting_strength,
                request.team_b_batting_strength,
                request.team_a_bowling_strength,
                request.team_b_bowling_strength,
                parsed_overrides,
            )

            request = request.model_copy(
                update={
                    "team_a_batting_strength": team_a_bat,
                    "team_b_batting_strength": team_b_bat,
                    "team_a_bowling_strength": team_a_bowl,
                    "team_b_bowling_strength": team_b_bowl,
                }
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
                ai_analysis="",
                feature_snapshot=request.model_dump(),
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

    def rebuild_upcoming_predictions(self) -> list[dict]:
        """Recompute saved future predictions using the latest live context."""
        self._invalidate_future_predictions()
        return self.predict_upcoming_matches()

    def check_results_and_learn(self) -> dict:
        """Check cricsheet for match results and trigger retraining if needed.

        Returns a summary dict with keys: checked, updated, retrained.
        """
        checked, updated = 0, 0

        pending_ids = self._db.get_pending_result_match_ids()
        if not pending_ids:
            return {"checked": 0, "updated": 0, "retrained": False}

        # Merge fast points-table results with cricsheet data; if both sources
        # know a result, prefer cricsheet because it is the richer scorecard feed.
        result_lookup = self._fetch_points_table_results()
        result_lookup.update(self._fetch_local_csv_results())
        result_lookup.update(self._fetch_cricsheet_results())

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
                updated += 1
                if winner == "No Result":
                    log.info("Result recorded for %s: abandoned / no result", match_id)
                elif was_correct is not None:
                    log.info(
                        "Result recorded for %s: actual=%s (%s)",
                        match_id, winner, "✓" if was_correct else "✗",
                    )

        retrained = False
        stats = self._db.get_accuracy_stats()
        completed_since_retrain = self._db.count_resolved_predictions_since(stats.get("last_retrain_at"))
        retrain_reason: str | None = None
        if (
            stats["wrong_since_retrain"] >= RETRAIN_WRONG_THRESHOLD
            and stats["total"] >= RETRAIN_MIN_PREDICTIONS
        ):
            retrain_reason = f"{stats['wrong_since_retrain']} wrong since last retrain"
        elif completed_since_retrain >= RETRAIN_FEEDBACK_THRESHOLD:
            retrain_reason = f"{completed_since_retrain} completed predictions since last retrain"

        if retrain_reason:
            log.info(
                "Triggering self-learning retrain (%s).",
                retrain_reason,
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
            if not (saved.get("ai_analysis") or "").strip():
                analysis = self.ensure_prediction_analysis(nxt["match_id"])
                if analysis:
                    saved["ai_analysis"] = analysis
            return {**nxt, **saved}
        return dict(nxt)

    def ensure_prediction_analysis(self, match_id: str) -> str | None:
        """Generate and persist AI analysis for a saved prediction on demand."""
        saved = self._db.get_prediction(match_id)
        if not saved:
            return None

        match = self._schedule.get_match_by_id(match_id)
        if match is None:
            existing = (saved.get("ai_analysis") or "").strip()
            return existing or None

        existing = (saved.get("ai_analysis") or "").strip()

        standings = get_standings_service()
        team_a = match["team_a"]
        team_b = match["team_b"]
        verified_context = self._build_verified_player_context(team_a, team_b)
        if existing and not self._analysis_needs_refresh(existing, team_a, team_b, match.get("venue", ""), verified_context):
            return existing

        ta_standing = standings.get_team(team_a)
        tb_standing = standings.get_team(team_b)
        ta_games = ta_standing.played if ta_standing else 0
        tb_games = tb_standing.played if tb_standing else 0
        team_a_form = standings.recent_form(team_a)
        team_b_form = standings.recent_form(team_b)
        team_a_bat = standings.batting_strength(team_a) if ta_games >= _MIN_GAMES_FOR_NRR else 65.0
        team_b_bat = standings.batting_strength(team_b) if tb_games >= _MIN_GAMES_FOR_NRR else 65.0
        team_a_bowl = standings.bowling_strength(team_a) if ta_games >= _MIN_GAMES_FOR_NRR else 65.0
        team_b_bowl = standings.bowling_strength(team_b) if tb_games >= _MIN_GAMES_FOR_NRR else 65.0

        active_overrides = self._db.get_active_overrides()
        parsed_overrides = [override["parsed"] for override in active_overrides]
        team_a_bat, team_b_bat, team_a_bowl, team_b_bowl = apply_overrides(
            team_a, team_b, team_a_bat, team_b_bat, team_a_bowl, team_b_bowl,
            parsed_overrides,
        )

        venue_adv = venue_advantage(match["venue"], team_a, team_b)
        winner = saved.get("predicted_winner", "")
        win_pct = max(
            float(saved.get("team_a_probability") or 0.5),
            float(saved.get("team_b_probability") or 0.5),
        ) * 100
        analysis = self._generate_ai_analysis(
            match, team_a, team_b, team_a_form, team_b_form,
            team_a_bat, team_b_bat, team_a_bowl, team_b_bowl,
            venue_adv, winner, win_pct, active_overrides, verified_context,
        )
        if analysis:
            self._db.update_prediction_analysis(match_id, analysis)
        return analysis

    def get_recent_history(self, limit: int = 10) -> list[dict]:
        return self._db.get_recent_predictions(limit)

    def get_paginated_history(self, page: int = 1, per_page: int = 10) -> tuple[list[dict], int]:
        return self._db.get_paginated_predictions(page, per_page)

    def get_accuracy_stats(self) -> dict:
        return self._db.get_accuracy_stats()

    # ------------------------------------------------------------------
    # LLM AI analysis
    # ------------------------------------------------------------------

    def _generate_ai_analysis(
        self,
        match: dict,
        team_a: str,
        team_b: str,
        ta_form: float,
        tb_form: float,
        ta_bat: float,
        tb_bat: float,
        ta_bowl: float,
        tb_bowl: float,
        venue_adv: float,
        predicted_winner: str,
        win_pct: float,
        active_overrides: list[dict],
        verified_context: dict[str, list[str]],
    ) -> str | None:
        """Call the configured LLM to generate a pre-match analysis."""
        from cricket_predictor.providers.gemini_provider import generate_match_analysis

        # Build injury summary from active overrides
        injury_notes = []
        other_notes = []
        for ov in active_overrides:
            p = ov.get("parsed", {})
            desc = p.get("description", ov.get("note", ""))
            if "injur" in desc.lower() or "unavailable" in desc.lower() or "miss" in desc.lower():
                injury_notes.append(desc)
            elif desc:
                other_notes.append(desc)

        context = {
            "team_a": team_a,
            "team_b": team_b,
            "venue": match.get("venue", ""),
            "match_date": match.get("match_date", ""),
            "team_a_batting": ta_bat,
            "team_b_batting": tb_bat,
            "team_a_bowling": ta_bowl,
            "team_b_bowling": tb_bowl,
            "team_a_form": ta_form,
            "team_b_form": tb_form,
            "venue_advantage": venue_adv,
            "predicted_winner": predicted_winner,
            "win_probability": win_pct,
            "injuries": "; ".join(injury_notes) if injury_notes else "None reported",
            "overrides": "; ".join(other_notes) if other_notes else "None",
        }
        context.update(verified_context)

        try:
            return generate_match_analysis(context)
        except Exception as exc:
            log.warning("AI analysis failed for %s: %s", match.get("match_id"), exc)
            return None

    def _build_verified_player_context(self, team_a: str, team_b: str) -> dict[str, list[str]]:
        empty = {
            "verified_team_a_squad": [],
            "verified_team_b_squad": [],
            "verified_team_a_batting_leaders": [],
            "verified_team_b_batting_leaders": [],
            "verified_team_a_bowling_leaders": [],
            "verified_team_b_bowling_leaders": [],
        }
        if not self._settings.ipl_csv_data_dir:
            return empty

        provider = IplCsvDataProvider(self._settings.ipl_csv_data_dir)
        squad_lookup = provider.team_squad_lookup()
        leader_lookup = provider.team_leader_names_lookup()
        team_a_leaders = leader_lookup.get(team_a)
        team_b_leaders = leader_lookup.get(team_b)
        return {
            "verified_team_a_squad": squad_lookup.get(team_a, []),
            "verified_team_b_squad": squad_lookup.get(team_b, []),
            "verified_team_a_batting_leaders": list(team_a_leaders.top_batters) if team_a_leaders else [],
            "verified_team_b_batting_leaders": list(team_b_leaders.top_batters) if team_b_leaders else [],
            "verified_team_a_bowling_leaders": list(team_a_leaders.top_bowlers) if team_a_leaders else [],
            "verified_team_b_bowling_leaders": list(team_b_leaders.top_bowlers) if team_b_leaders else [],
        }

    def _analysis_needs_refresh(
        self,
        existing_analysis: str,
        team_a: str,
        team_b: str,
        venue: str,
        verified_context: dict[str, list[str]],
    ) -> bool:
        allowed_full_names = {
            self._normalise_text(name)
            for key, names in verified_context.items()
            if key.startswith("verified_")
            for name in names
            if name
        }
        if not allowed_full_names:
            return False

        allowed_tokens = {
            token
            for name in allowed_full_names
            for token in name.split()
            if len(token) > 2
        }
        allowed_full_names.update(
            self._normalise_text(value)
            for value in (team_a, team_b, venue)
            if value
        )
        allowed_tokens.update(
            self._normalise_text(token)
            for value in (team_a, team_b, venue)
            for token in str(value).split()
            if len(token) > 2
        )

        for match in _PLAYER_MENTION_PATTERN.finditer(existing_analysis):
            fragment = match.group(1)
            for part in re.split(r",|\band\b", fragment):
                candidate = self._normalise_text(part)
                if not candidate:
                    continue
                if candidate in allowed_full_names:
                    continue
                if all(token in allowed_tokens for token in candidate.split()):
                    continue
                return True
        return False

    def _normalise_text(self, value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))

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
                    # Abandoned / no result / tie without a super-over winner
                    result_type = outcome.get("result", "")
                    if result_type in ("no result", "tie") or not winner_raw:
                        team_a_raw, team_b_raw = teams
                        team_a = resolve_team_name(SHORT_TEAM.get(team_a_raw, team_a_raw))
                        team_b = resolve_team_name(SHORT_TEAM.get(team_b_raw, team_b_raw))
                        if result_type in ("no result",):
                            results[(team_a, team_b, dates[0])] = "No Result"
                            results[(team_b, team_a, dates[0])] = "No Result"
                        continue
                # Normalise team name to full name via SHORT_TEAM aliases
                winner = resolve_team_name(SHORT_TEAM.get(winner_raw, winner_raw))
                team_a_raw, team_b_raw = teams
                team_a = resolve_team_name(SHORT_TEAM.get(team_a_raw, team_a_raw))
                team_b = resolve_team_name(SHORT_TEAM.get(team_b_raw, team_b_raw))
                results[(team_a, team_b, dates[0])] = winner
                results[(team_b, team_a, dates[0])] = winner
            except Exception:  # noqa: BLE001
                continue
        return results

    def _fetch_points_table_results(self) -> dict[tuple[str, str, str], str]:
        """Extract recent completed results from the live points-table page."""
        standings = get_standings_service()
        cached_lookup = standings.recent_results_lookup()
        if cached_lookup:
            return cached_lookup

        try:
            provider = CricinfoStandingsProvider(self._settings.cricinfo_standings_url)
            return build_recent_results_lookup(provider.fetch_recent_results())
        except Exception as exc:  # noqa: BLE001
            log.warning("Points-table result fetch failed: %s", exc)
            return {}

    def _fetch_local_csv_results(self) -> dict[tuple[str, str, str], str]:
        """Extract completed winners from a local IPL CSV export when configured."""
        dataset_dir = self._settings.ipl_csv_data_dir
        if not dataset_dir:
            return {}

        try:
            return IplCsvDataProvider(dataset_dir).fetch_results_lookup()
        except Exception as exc:  # noqa: BLE001
            log.warning("Local IPL CSV result fetch failed: %s", exc)
            return {}

    def _do_retrain(self) -> bool:
        """Retrain from local cricsheet data plus completed prediction history."""
        try:
            from cricket_predictor.services.data_update_service import DataUpdateService
            from cricket_predictor.services.prediction_service import get_prediction_service

            svc = DataUpdateService(self._settings)
            retrained = svc.retrain_from_local_data()
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
