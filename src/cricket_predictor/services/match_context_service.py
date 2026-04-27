from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from cricket_predictor.api.schemas import MatchPredictionRequest
from cricket_predictor.config.settings import Settings, get_settings
from cricket_predictor.providers.cricinfo_standings import resolve_team_name, venue_advantage
from cricket_predictor.providers.ipl_csv_provider import IplCsvDataProvider, TeamLeaderStats, TeamMetrics
from cricket_predictor.providers.iplt20_stats_provider import fetch_team_leader_stats
from cricket_predictor.providers.match_history_provider import MatchHistoryProvider
from cricket_predictor.services.standings_service import get_standings_service

# Minimum games a team must have played before we trust standings-derived strengths.
_MIN_GAMES_FOR_NRR = 1


@dataclass(frozen=True)
class MatchContextSignals:
    team_a_recent_form: float
    team_b_recent_form: float
    team_a_batting_strength: float
    team_b_batting_strength: float
    team_a_bowling_strength: float
    team_b_bowling_strength: float
    head_to_head_win_pct_team_a: float
    team_a_top_run_getters_runs: float
    team_b_top_run_getters_runs: float
    team_a_top_wicket_takers_wickets: float
    team_b_top_wicket_takers_wickets: float


class MatchContextService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._history = MatchHistoryProvider(settings.cricsheet_data_dir)
        self._csv_provider = (
            IplCsvDataProvider(settings.ipl_csv_data_dir)
            if settings.ipl_csv_data_dir
            else None
        )

    def build_request(
        self,
        *,
        team_a: str,
        team_b: str,
        venue: str,
        match_format: str,
        pitch_type: str,
        toss_winner: str,
        toss_decision: str,
        dew_probability: float = 0.3,
        pitch_batting_bias: float = 0.0,
        night_match: bool = True,
    ) -> MatchPredictionRequest:
        canonical_a = resolve_team_name(team_a)
        canonical_b = resolve_team_name(team_b)
        signals = self.get_signals(canonical_a, canonical_b)

        return MatchPredictionRequest(
            team_a=canonical_a,
            team_b=canonical_b,
            venue=venue,
            match_format=match_format,
            pitch_type=pitch_type,
            toss_winner=resolve_team_name(toss_winner),
            toss_decision=toss_decision,
            team_a_recent_form=signals.team_a_recent_form,
            team_b_recent_form=signals.team_b_recent_form,
            team_a_batting_strength=signals.team_a_batting_strength,
            team_b_batting_strength=signals.team_b_batting_strength,
            team_a_bowling_strength=signals.team_a_bowling_strength,
            team_b_bowling_strength=signals.team_b_bowling_strength,
            head_to_head_win_pct_team_a=signals.head_to_head_win_pct_team_a,
            venue_advantage_team_a=venue_advantage(venue, canonical_a, canonical_b),
            team_a_top_run_getters_runs=signals.team_a_top_run_getters_runs,
            team_b_top_run_getters_runs=signals.team_b_top_run_getters_runs,
            team_a_top_wicket_takers_wickets=signals.team_a_top_wicket_takers_wickets,
            team_b_top_wicket_takers_wickets=signals.team_b_top_wicket_takers_wickets,
            dew_probability=dew_probability,
            pitch_batting_bias=pitch_batting_bias,
            night_match=night_match,
        )

    def get_signals(self, team_a: str, team_b: str) -> MatchContextSignals:
        standings = get_standings_service()
        ta_standing = standings.get_team(team_a)
        tb_standing = standings.get_team(team_b)
        ta_games = ta_standing.played if ta_standing else 0
        tb_games = tb_standing.played if tb_standing else 0

        # Prefer current-season standings form; fall back to cricsheet history
        # only when the team is absent from the standings table.
        team_a_recent_form = standings.recent_form(team_a) if ta_standing else self._history.recent_form(team_a)
        team_b_recent_form = standings.recent_form(team_b) if tb_standing else self._history.recent_form(team_b)
        team_a_batting_strength = standings.batting_strength(team_a) if ta_games >= _MIN_GAMES_FOR_NRR else 65.0
        team_b_batting_strength = standings.batting_strength(team_b) if tb_games >= _MIN_GAMES_FOR_NRR else 65.0
        team_a_bowling_strength = standings.bowling_strength(team_a) if ta_games >= _MIN_GAMES_FOR_NRR else 65.0
        team_b_bowling_strength = standings.bowling_strength(team_b) if tb_games >= _MIN_GAMES_FOR_NRR else 65.0
        head_to_head = self._history.head_to_head_pct(team_a, team_b, limit=7, fallback=0.5)
        team_a_leaders = TeamLeaderStats()
        team_b_leaders = TeamLeaderStats()

        # --- Leader stats: iplt20.com S3 feeds first, Kaggle CSV fallback ---
        iplt20_leaders = fetch_team_leader_stats(self._settings.iplt20_stats_competition_id)
        if iplt20_leaders:
            team_a_leaders = iplt20_leaders.get(team_a, team_a_leaders)
            team_b_leaders = iplt20_leaders.get(team_b, team_b_leaders)

        if self._csv_provider is not None:
            metrics = self._csv_provider.team_metrics_lookup()
            team_a_metrics = metrics.get(team_a)
            team_b_metrics = metrics.get(team_b)
            if team_a_metrics is not None:
                team_a_recent_form = team_a_metrics.recent_form_pct
                team_a_batting_strength = team_a_metrics.batting_strength
                team_a_bowling_strength = team_a_metrics.bowling_strength
            if team_b_metrics is not None:
                team_b_recent_form = team_b_metrics.recent_form_pct
                team_b_batting_strength = team_b_metrics.batting_strength
                team_b_bowling_strength = team_b_metrics.bowling_strength

            csv_h2h = self._csv_provider.head_to_head_pct(team_a, team_b, limit=7)
            if head_to_head == 0.5 and csv_h2h != 0.5:
                head_to_head = csv_h2h

            # Fall back to Kaggle CSV leaders only when iplt20 feeds failed
            if not iplt20_leaders:
                csv_leaders = self._csv_provider.team_leader_stats_lookup()
                team_a_leaders = csv_leaders.get(team_a, team_a_leaders)
                team_b_leaders = csv_leaders.get(team_b, team_b_leaders)

        return MatchContextSignals(
            team_a_recent_form=team_a_recent_form,
            team_b_recent_form=team_b_recent_form,
            team_a_batting_strength=team_a_batting_strength,
            team_b_batting_strength=team_b_batting_strength,
            team_a_bowling_strength=team_a_bowling_strength,
            team_b_bowling_strength=team_b_bowling_strength,
            head_to_head_win_pct_team_a=head_to_head,
            team_a_top_run_getters_runs=team_a_leaders.top_run_getters_runs,
            team_b_top_run_getters_runs=team_b_leaders.top_run_getters_runs,
            team_a_top_wicket_takers_wickets=team_a_leaders.top_wicket_takers_wickets,
            team_b_top_wicket_takers_wickets=team_b_leaders.top_wicket_takers_wickets,
        )


@lru_cache
def _get_default_match_context_service() -> MatchContextService:
    return MatchContextService(get_settings())


def get_match_context_service(settings: Settings | None = None) -> MatchContextService:
    if settings is None:
        return _get_default_match_context_service()
    return MatchContextService(settings)