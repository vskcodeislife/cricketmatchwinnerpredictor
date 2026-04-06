from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from cricket_predictor.providers.base import LiveDataProvider
from cricket_predictor.providers.cricinfo_standings import resolve_team_name, venue_advantage

log = logging.getLogger(__name__)

_TEAM_A_COLUMNS = ("team_a", "teama", "team_1", "team1", "home_team", "first_team")
_TEAM_B_COLUMNS = ("team_b", "teamb", "team_2", "team2", "away_team", "second_team")
_TEAM_COLUMNS = ("team", "team_name", "teamname", "short_name", "shortname", "club")
_MATCH_ID_COLUMNS = ("match_id", "matchid", "id")
_MATCH_DATE_COLUMNS = ("match_date", "date", "start_date", "fixture_date", "scheduled_date")
_WINNER_COLUMNS = ("winner", "winning_team", "match_winner", "result_winner")
_STATUS_COLUMNS = ("status", "match_status", "state", "result")
_VENUE_COLUMNS = ("venue", "ground", "stadium")
_TOSS_WINNER_COLUMNS = ("toss_winner", "tosswinningteam")
_TOSS_DECISION_COLUMNS = ("toss_decision", "tossdecision", "decision")
_PLAYED_COLUMNS = ("played", "pld", "matches", "p")
_WON_COLUMNS = ("won", "wins", "w")
_FORM_COLUMNS = ("form", "recent_form", "form_guide")
_INNINGS_COLUMNS = ("inning", "innings", "innings_no", "innings_number")
_BATTING_TEAM_COLUMNS = ("batting_team", "battingteam")
_BOWLING_TEAM_COLUMNS = ("bowling_team", "bowlingteam")
_TOTAL_RUNS_COLUMNS = ("total_runs", "totalrun", "runs_total")
_BATSMAN_RUNS_COLUMNS = ("batsman_runs", "batter_runs")
_EXTRA_RUNS_COLUMNS = ("extra_runs", "extras")


@dataclass(frozen=True)
class TeamMetrics:
    recent_form_pct: float = 0.5
    batting_strength: float = 65.0
    bowling_strength: float = 65.0


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalised = frame.copy()
    normalised.columns = [
        "".join(ch if ch.isalnum() else "_" for ch in str(column).strip().lower()).strip("_")
        for column in normalised.columns
    ]
    return normalised


def _first_existing(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _parse_date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.date().isoformat()


def _scale_batting(avg_runs: float) -> float:
    avg_runs = max(100.0, min(230.0, avg_runs))
    return round(40.0 + (avg_runs - 100.0) / 130.0 * 50.0, 2)


def _scale_bowling(avg_runs: float) -> float:
    avg_runs = max(100.0, min(230.0, avg_runs))
    return round(40.0 + (230.0 - avg_runs) / 130.0 * 50.0, 2)


class IplCsvDataProvider(LiveDataProvider):
    def __init__(self, dataset_dir: str | Path) -> None:
        self._dataset_dir = Path(dataset_dir)

    async def fetch_live_match_context(self) -> list[dict[str, Any]]:
        matches = self._load_csv("matches.csv")
        if matches.empty:
            return []

        team_metrics = self._build_team_metrics(matches)
        head_to_head = self._build_head_to_head_lookup(matches)

        team_a_col = _first_existing(matches, _TEAM_A_COLUMNS)
        team_b_col = _first_existing(matches, _TEAM_B_COLUMNS)
        date_col = _first_existing(matches, _MATCH_DATE_COLUMNS)
        venue_col = _first_existing(matches, _VENUE_COLUMNS)
        winner_col = _first_existing(matches, _WINNER_COLUMNS)
        status_col = _first_existing(matches, _STATUS_COLUMNS)
        toss_winner_col = _first_existing(matches, _TOSS_WINNER_COLUMNS)
        toss_decision_col = _first_existing(matches, _TOSS_DECISION_COLUMNS)
        if not (team_a_col and team_b_col and date_col and venue_col):
            return []

        contexts: list[dict[str, Any]] = []
        today = date.today().isoformat()

        for _, row in matches.iterrows():
            team_a = resolve_team_name(_clean_text(row.get(team_a_col)))
            team_b = resolve_team_name(_clean_text(row.get(team_b_col)))
            if not team_a or not team_b:
                continue

            match_date = _parse_date(row.get(date_col))
            winner = resolve_team_name(_clean_text(row.get(winner_col))) if winner_col else ""
            status = _clean_text(row.get(status_col)).lower() if status_col else ""
            if not self._is_pending_match(match_date, winner, status, today):
                continue

            team_a_metrics = team_metrics.get(team_a, TeamMetrics())
            team_b_metrics = team_metrics.get(team_b, TeamMetrics())
            toss_winner = resolve_team_name(_clean_text(row.get(toss_winner_col))) if toss_winner_col else team_a
            toss_decision = _clean_text(row.get(toss_decision_col)).lower() if toss_decision_col else ""
            if toss_decision not in {"bat", "bowl"}:
                toss_decision = "bat"

            contexts.append(
                {
                    "team_a": team_a,
                    "team_b": team_b,
                    "venue": _clean_text(row.get(venue_col)),
                    "match_format": "T20",
                    "pitch_type": "balanced",
                    "toss_winner": toss_winner or team_a,
                    "toss_decision": toss_decision,
                    "team_a_recent_form": team_a_metrics.recent_form_pct,
                    "team_b_recent_form": team_b_metrics.recent_form_pct,
                    "team_a_batting_strength": team_a_metrics.batting_strength,
                    "team_b_batting_strength": team_b_metrics.batting_strength,
                    "team_a_bowling_strength": team_a_metrics.bowling_strength,
                    "team_b_bowling_strength": team_b_metrics.bowling_strength,
                    "head_to_head_win_pct_team_a": self._head_to_head_pct(head_to_head, team_a, team_b),
                    "venue_advantage_team_a": venue_advantage(_clean_text(row.get(venue_col)), team_a, team_b),
                    "night_match": True,
                }
            )

        return contexts

    def fetch_results_lookup(self) -> dict[tuple[str, str, str], str]:
        matches = self._load_csv("matches.csv")
        if matches.empty:
            return {}

        team_a_col = _first_existing(matches, _TEAM_A_COLUMNS)
        team_b_col = _first_existing(matches, _TEAM_B_COLUMNS)
        date_col = _first_existing(matches, _MATCH_DATE_COLUMNS)
        winner_col = _first_existing(matches, _WINNER_COLUMNS)
        if not (team_a_col and team_b_col and date_col and winner_col):
            return {}

        results: dict[tuple[str, str, str], str] = {}
        for _, row in matches.iterrows():
            team_a = resolve_team_name(_clean_text(row.get(team_a_col)))
            team_b = resolve_team_name(_clean_text(row.get(team_b_col)))
            winner = resolve_team_name(_clean_text(row.get(winner_col)))
            match_date = _parse_date(row.get(date_col))
            if not (team_a and team_b and winner and match_date):
                continue
            results[(team_a, team_b, match_date)] = winner
            results[(team_b, team_a, match_date)] = winner
        return results

    def _load_csv(self, filename: str) -> pd.DataFrame:
        path = self._dataset_dir / filename
        if not path.exists():
            return pd.DataFrame()
        try:
            return _normalise_columns(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _build_team_metrics(self, matches: pd.DataFrame) -> dict[str, TeamMetrics]:
        points_table = self._load_csv("points_table.csv")
        deliveries = self._load_csv("deliveries.csv")

        recent_form = self._recent_form_from_points_table(points_table)
        if not recent_form:
            recent_form = self._recent_form_from_matches(matches)
        strengths = self._strengths_from_deliveries(deliveries)

        all_teams = set(recent_form) | set(strengths)
        team_a_col = _first_existing(matches, _TEAM_A_COLUMNS)
        team_b_col = _first_existing(matches, _TEAM_B_COLUMNS)
        if team_a_col and team_b_col:
            for _, row in matches.iterrows():
                team_a = resolve_team_name(_clean_text(row.get(team_a_col)))
                team_b = resolve_team_name(_clean_text(row.get(team_b_col)))
                if team_a:
                    all_teams.add(team_a)
                if team_b:
                    all_teams.add(team_b)

        return {
            team: TeamMetrics(
                recent_form_pct=recent_form.get(team, 0.5),
                batting_strength=strengths.get(team, (65.0, 65.0))[0],
                bowling_strength=strengths.get(team, (65.0, 65.0))[1],
            )
            for team in all_teams
        }

    def _recent_form_from_points_table(self, points_table: pd.DataFrame) -> dict[str, float]:
        if points_table.empty:
            return {}

        team_col = _first_existing(points_table, _TEAM_COLUMNS)
        played_col = _first_existing(points_table, _PLAYED_COLUMNS)
        won_col = _first_existing(points_table, _WON_COLUMNS)
        form_col = _first_existing(points_table, _FORM_COLUMNS)
        if team_col is None:
            return {}

        result: dict[str, float] = {}
        for _, row in points_table.iterrows():
            team = resolve_team_name(_clean_text(row.get(team_col)))
            if not team:
                continue

            form_text = _clean_text(row.get(form_col)) if form_col else ""
            if form_text:
                tokens = [token for token in form_text.replace("-", " ").split() if token in {"W", "L"}]
                if tokens:
                    result[team] = round(tokens.count("W") / len(tokens), 3)
                    continue

            played = float(row.get(played_col) or 0) if played_col else 0.0
            won = float(row.get(won_col) or 0) if won_col else 0.0
            result[team] = round(won / played, 3) if played > 0 else 0.5
        return result

    def _recent_form_from_matches(self, matches: pd.DataFrame) -> dict[str, float]:
        team_a_col = _first_existing(matches, _TEAM_A_COLUMNS)
        team_b_col = _first_existing(matches, _TEAM_B_COLUMNS)
        date_col = _first_existing(matches, _MATCH_DATE_COLUMNS)
        winner_col = _first_existing(matches, _WINNER_COLUMNS)
        if not (team_a_col and team_b_col and date_col and winner_col):
            return {}

        completed: list[tuple[str, str, str, str]] = []
        for _, row in matches.iterrows():
            team_a = resolve_team_name(_clean_text(row.get(team_a_col)))
            team_b = resolve_team_name(_clean_text(row.get(team_b_col)))
            winner = resolve_team_name(_clean_text(row.get(winner_col)))
            match_date = _parse_date(row.get(date_col))
            if team_a and team_b and winner and match_date:
                completed.append((match_date, team_a, team_b, winner))

        completed.sort(key=lambda item: item[0])
        team_results: dict[str, list[int]] = defaultdict(list)
        for _, team_a, team_b, winner in completed:
            team_results[team_a].append(int(winner == team_a))
            team_results[team_b].append(int(winner == team_b))

        return {
            team: round(sum(results[-5:]) / len(results[-5:]), 3)
            for team, results in team_results.items()
            if results
        }

    def _strengths_from_deliveries(self, deliveries: pd.DataFrame) -> dict[str, tuple[float, float]]:
        if deliveries.empty:
            return {}

        batting_team_col = _first_existing(deliveries, _BATTING_TEAM_COLUMNS)
        bowling_team_col = _first_existing(deliveries, _BOWLING_TEAM_COLUMNS)
        match_id_col = _first_existing(deliveries, _MATCH_ID_COLUMNS)
        innings_col = _first_existing(deliveries, _INNINGS_COLUMNS)
        total_runs_col = _first_existing(deliveries, _TOTAL_RUNS_COLUMNS)
        if total_runs_col is None:
            batsman_runs_col = _first_existing(deliveries, _BATSMAN_RUNS_COLUMNS)
            extra_runs_col = _first_existing(deliveries, _EXTRA_RUNS_COLUMNS)
            if batsman_runs_col and extra_runs_col:
                deliveries = deliveries.copy()
                deliveries["_computed_total_runs"] = deliveries[batsman_runs_col] + deliveries[extra_runs_col]
                total_runs_col = "_computed_total_runs"

        if not (batting_team_col and bowling_team_col and match_id_col and total_runs_col):
            return {}

        group_columns = [match_id_col, batting_team_col, bowling_team_col]
        if innings_col:
            group_columns.insert(1, innings_col)
        innings_totals = (
            deliveries.groupby(group_columns, dropna=False)[total_runs_col].sum().reset_index()
        )

        scored: dict[str, list[float]] = defaultdict(list)
        conceded: dict[str, list[float]] = defaultdict(list)
        for _, row in innings_totals.iterrows():
            batting_team = resolve_team_name(_clean_text(row.get(batting_team_col)))
            bowling_team = resolve_team_name(_clean_text(row.get(bowling_team_col)))
            runs = float(row.get(total_runs_col) or 0.0)
            if batting_team:
                scored[batting_team].append(runs)
            if bowling_team:
                conceded[bowling_team].append(runs)

        strengths: dict[str, tuple[float, float]] = {}
        for team in set(scored) | set(conceded):
            avg_scored = sum(scored.get(team, [165.0])) / max(len(scored.get(team, [])), 1)
            avg_conceded = sum(conceded.get(team, [165.0])) / max(len(conceded.get(team, [])), 1)
            strengths[team] = (_scale_batting(avg_scored), _scale_bowling(avg_conceded))
        return strengths

    def _build_head_to_head_lookup(self, matches: pd.DataFrame) -> dict[frozenset[str], dict[str, int]]:
        team_a_col = _first_existing(matches, _TEAM_A_COLUMNS)
        team_b_col = _first_existing(matches, _TEAM_B_COLUMNS)
        winner_col = _first_existing(matches, _WINNER_COLUMNS)
        if not (team_a_col and team_b_col and winner_col):
            return {}

        lookup: dict[frozenset[str], dict[str, int]] = {}
        for _, row in matches.iterrows():
            team_a = resolve_team_name(_clean_text(row.get(team_a_col)))
            team_b = resolve_team_name(_clean_text(row.get(team_b_col)))
            winner = resolve_team_name(_clean_text(row.get(winner_col)))
            if not (team_a and team_b and winner):
                continue
            key = frozenset((team_a, team_b))
            counts = lookup.setdefault(key, {"total": 0})
            counts["total"] += 1
            counts[winner] = counts.get(winner, 0) + 1
        return lookup

    def _head_to_head_pct(
        self,
        lookup: dict[frozenset[str], dict[str, int]],
        team_a: str,
        team_b: str,
    ) -> float:
        counts = lookup.get(frozenset((team_a, team_b)))
        if not counts or counts.get("total", 0) == 0:
            return 0.5
        return round(counts.get(team_a, 0) / counts["total"], 3)

    def _is_pending_match(self, match_date: str, winner: str, status: str, today: str) -> bool:
        if winner:
            return False
        if any(keyword in status for keyword in ("live", "upcoming", "scheduled", "preview", "toss")):
            return True
        if status and any(keyword in status for keyword in ("completed", "result", "won by", "abandoned")):
            return False
        return bool(match_date) and match_date >= today