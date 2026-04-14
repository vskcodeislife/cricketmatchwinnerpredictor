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
_PLAYER_COLUMNS = ("player", "player_name", "name", "full_name", "batter", "batsman", "bowler")
_MATCH_ID_COLUMNS = ("match_id", "matchid", "id", "match_no")
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
_BATSMAN_RUNS_COLUMNS = ("batsman_runs", "batter_runs", "runs_of_bat", "runs_off_bat")
_EXTRA_RUNS_COLUMNS = ("extra_runs", "extras")
_BATTER_COLUMNS = ("batter", "batsman", "striker")
_BOWLER_COLUMNS = ("bowler",)
_PLAYER_DISMISSED_COLUMNS = ("player_dismissed", "dismissed_player")
_DISMISSAL_KIND_COLUMNS = ("dismissal_kind", "wicket_type")
_IS_WICKET_COLUMNS = ("is_wicket", "wicket")
_RUNS_COLUMNS = ("runs", "total_runs", "batting_runs", "orange_cap_runs")
_WICKETS_COLUMNS = ("wickets", "total_wickets", "wkts", "purple_cap_wickets")

_CSV_USECOLS = {
    "matches.csv": set(
        _TEAM_A_COLUMNS
        + _TEAM_B_COLUMNS
        + _MATCH_DATE_COLUMNS
        + _WINNER_COLUMNS
        + _STATUS_COLUMNS
        + _VENUE_COLUMNS
        + _TOSS_WINNER_COLUMNS
        + _TOSS_DECISION_COLUMNS
    ),
    "points_table.csv": set(_TEAM_COLUMNS + _PLAYED_COLUMNS + _WON_COLUMNS + _FORM_COLUMNS),
    "orange_cap.csv": set(_TEAM_COLUMNS + _PLAYER_COLUMNS + _RUNS_COLUMNS),
    "purple_cap.csv": set(_TEAM_COLUMNS + _PLAYER_COLUMNS + _WICKETS_COLUMNS),
    "squads.csv": set(_TEAM_COLUMNS + _PLAYER_COLUMNS),
    "deliveries.csv": set(
        _MATCH_ID_COLUMNS
        + _INNINGS_COLUMNS
        + _BATTING_TEAM_COLUMNS
        + _BOWLING_TEAM_COLUMNS
        + _TOTAL_RUNS_COLUMNS
        + _BATSMAN_RUNS_COLUMNS
        + _EXTRA_RUNS_COLUMNS
        + _BATTER_COLUMNS
        + _BOWLER_COLUMNS
        + _PLAYER_DISMISSED_COLUMNS
        + _DISMISSAL_KIND_COLUMNS
        + _IS_WICKET_COLUMNS
    ),
}

_NON_BOWLER_DISMISSALS = {"run out", "retired hurt", "retired out", "obstructing the field"}


@dataclass(frozen=True)
class TeamMetrics:
    recent_form_pct: float = 0.5
    batting_strength: float = 65.0
    bowling_strength: float = 65.0


@dataclass(frozen=True)
class TeamLeaderStats:
    top_run_getters_runs: float = 0.0
    top_wicket_takers_wickets: float = 0.0


@dataclass(frozen=True)
class TeamLeaderNames:
    top_batters: tuple[str, ...] = ()
    top_bowlers: tuple[str, ...] = ()


def _normalise_column_name(column: object) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(column).strip().lower()).strip("_")


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalised = frame.copy()
    normalised.columns = [_normalise_column_name(column) for column in normalised.columns]
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


def _player_key(value: object) -> str:
    return _clean_text(value).lower()


def _coerce_float(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


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
        team_leaders = self.team_leader_stats_lookup()
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
            team_a_leaders = team_leaders.get(team_a, TeamLeaderStats())
            team_b_leaders = team_leaders.get(team_b, TeamLeaderStats())
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
                    "head_to_head_win_pct_team_a": self._head_to_head_pct(head_to_head, team_a, team_b, limit=7),
                    "venue_advantage_team_a": venue_advantage(_clean_text(row.get(venue_col)), team_a, team_b),
                    "team_a_top_run_getters_runs": team_a_leaders.top_run_getters_runs,
                    "team_b_top_run_getters_runs": team_b_leaders.top_run_getters_runs,
                    "team_a_top_wicket_takers_wickets": team_a_leaders.top_wicket_takers_wickets,
                    "team_b_top_wicket_takers_wickets": team_b_leaders.top_wicket_takers_wickets,
                    "night_match": True,
                }
            )

        return contexts

    def team_metrics_lookup(self) -> dict[str, TeamMetrics]:
        matches = self._load_csv("matches.csv")
        if matches.empty:
            return {}
        return self._build_team_metrics(matches)

    def team_squad_lookup(self) -> dict[str, list[str]]:
        squads = self._load_csv("squads.csv")
        if squads.empty:
            return {}

        team_col = _first_existing(squads, _TEAM_COLUMNS)
        player_col = _first_existing(squads, _PLAYER_COLUMNS)
        if not (team_col and player_col):
            return {}

        squads_by_team: dict[str, list[str]] = defaultdict(list)
        seen: dict[str, set[str]] = defaultdict(set)
        for _, row in squads.iterrows():
            team = resolve_team_name(_clean_text(row.get(team_col)))
            player_name = _clean_text(row.get(player_col))
            player_key = _player_key(player_name)
            if not team or not player_name or player_key in seen[team]:
                continue
            seen[team].add(player_key)
            squads_by_team[team].append(player_name)
        return dict(squads_by_team)

    def team_leader_stats_lookup(self) -> dict[str, TeamLeaderStats]:
        cap_leaders = self._build_team_leader_stats_from_caps(
            orange_cap=self._load_csv("orange_cap.csv"),
            purple_cap=self._load_csv("purple_cap.csv"),
            squads=self._load_csv("squads.csv"),
        )
        delivery_leaders = self._build_team_leader_stats(self._load_csv("deliveries.csv"))
        return self._merge_team_leader_stats(cap_leaders, delivery_leaders)

    def team_leader_names_lookup(self) -> dict[str, TeamLeaderNames]:
        squads = self._load_csv("squads.csv")
        if squads.empty:
            return {}
        player_teams = self._build_player_team_lookup(squads)
        return self._build_team_leader_names_from_caps(
            orange_cap=self._load_csv("orange_cap.csv"),
            purple_cap=self._load_csv("purple_cap.csv"),
            player_teams=player_teams,
        )

    def head_to_head_pct(self, team_a: str, team_b: str, limit: int = 7) -> float:
        matches = self._load_csv("matches.csv")
        if matches.empty:
            return 0.5
        return self._head_to_head_pct(
            self._build_head_to_head_lookup(matches),
            resolve_team_name(team_a),
            resolve_team_name(team_b),
            limit=limit,
        )

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
        read_kwargs: dict[str, Any] = {}
        selected_columns = _CSV_USECOLS.get(filename)
        if selected_columns:
            read_kwargs["usecols"] = lambda column: _normalise_column_name(column) in selected_columns
        try:
            return _normalise_columns(pd.read_csv(path, **read_kwargs))
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _build_team_metrics(self, matches: pd.DataFrame) -> dict[str, TeamMetrics]:
        points_table = self._load_csv("points_table.csv")
        deliveries = self._load_csv("deliveries.csv")

        recent_form = self._recent_form_from_points_table(points_table)
        if not recent_form:
            recent_form = self._recent_form_from_matches(matches)
        else:
            for team, form_value in self._recent_form_from_matches(matches).items():
                recent_form.setdefault(team, form_value)
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

    def _build_team_leader_stats(self, deliveries: pd.DataFrame) -> dict[str, TeamLeaderStats]:
        if deliveries.empty:
            return {}

        batting_team_col = _first_existing(deliveries, _BATTING_TEAM_COLUMNS)
        batter_col = _first_existing(deliveries, _BATTER_COLUMNS)
        batsman_runs_col = _first_existing(deliveries, _BATSMAN_RUNS_COLUMNS)
        bowling_team_col = _first_existing(deliveries, _BOWLING_TEAM_COLUMNS)
        bowler_col = _first_existing(deliveries, _BOWLER_COLUMNS)
        dismissal_kind_col = _first_existing(deliveries, _DISMISSAL_KIND_COLUMNS)
        player_dismissed_col = _first_existing(deliveries, _PLAYER_DISMISSED_COLUMNS)
        is_wicket_col = _first_existing(deliveries, _IS_WICKET_COLUMNS)

        leaders: dict[str, TeamLeaderStats] = {}

        if batting_team_col and batter_col and batsman_runs_col:
            batter_totals = (
                deliveries.groupby([batting_team_col, batter_col], dropna=False)[batsman_runs_col]
                .sum()
                .reset_index()
            )
            team_run_totals: dict[str, list[float]] = defaultdict(list)
            for _, row in batter_totals.iterrows():
                team = resolve_team_name(_clean_text(row.get(batting_team_col)))
                runs = float(row.get(batsman_runs_col) or 0.0)
                if team:
                    team_run_totals[team].append(runs)
            for team, values in team_run_totals.items():
                top_runs = sum(sorted(values, reverse=True)[:3])
                leaders[team] = TeamLeaderStats(top_run_getters_runs=round(top_runs, 2))

        if bowling_team_col and bowler_col and (is_wicket_col or player_dismissed_col):
            wickets_frame = deliveries.copy()
            if is_wicket_col:
                wickets_frame["_bowler_wicket"] = wickets_frame[is_wicket_col].fillna(0).astype(float)
            else:
                wickets_frame["_bowler_wicket"] = wickets_frame[player_dismissed_col].notna().astype(float)
            if dismissal_kind_col:
                wickets_frame.loc[
                    wickets_frame[dismissal_kind_col].fillna("").astype(str).str.lower().isin(_NON_BOWLER_DISMISSALS),
                    "_bowler_wicket",
                ] = 0.0

            bowler_totals = (
                wickets_frame.groupby([bowling_team_col, bowler_col], dropna=False)["_bowler_wicket"]
                .sum()
                .reset_index()
            )
            team_wicket_totals: dict[str, list[float]] = defaultdict(list)
            for _, row in bowler_totals.iterrows():
                team = resolve_team_name(_clean_text(row.get(bowling_team_col)))
                wickets = float(row.get("_bowler_wicket") or 0.0)
                if team:
                    team_wicket_totals[team].append(wickets)
            for team, values in team_wicket_totals.items():
                top_wickets = sum(sorted(values, reverse=True)[:3])
                current = leaders.get(team, TeamLeaderStats())
                leaders[team] = TeamLeaderStats(
                    top_run_getters_runs=current.top_run_getters_runs,
                    top_wicket_takers_wickets=round(top_wickets, 2),
                )

        return leaders

    def _build_team_leader_stats_from_caps(
        self,
        *,
        orange_cap: pd.DataFrame,
        purple_cap: pd.DataFrame,
        squads: pd.DataFrame,
    ) -> dict[str, TeamLeaderStats]:
        leaders: dict[str, TeamLeaderStats] = {}
        player_teams = self._build_player_team_lookup(squads)

        if not orange_cap.empty:
            team_col = _first_existing(orange_cap, _TEAM_COLUMNS + _BATTING_TEAM_COLUMNS)
            player_col = _first_existing(orange_cap, _PLAYER_COLUMNS)
            runs_col = _first_existing(orange_cap, _RUNS_COLUMNS)
            if runs_col and (team_col or player_col):
                team_run_totals: dict[str, dict[str, float]] = defaultdict(dict)
                for index, row in orange_cap.iterrows():
                    team = self._resolve_team_from_caps_row(row, team_col, player_col, player_teams)
                    runs = _coerce_float(row.get(runs_col))
                    if not team or runs is None:
                        continue
                    player = _player_key(row.get(player_col)) if player_col else str(index)
                    team_run_totals[team][player or str(index)] = max(team_run_totals[team].get(player or str(index), 0.0), runs)

                for team, values in team_run_totals.items():
                    leaders[team] = TeamLeaderStats(top_run_getters_runs=round(sum(sorted(values.values(), reverse=True)[:3]), 2))

        if not purple_cap.empty:
            team_col = _first_existing(purple_cap, _TEAM_COLUMNS + _BOWLING_TEAM_COLUMNS)
            player_col = _first_existing(purple_cap, _PLAYER_COLUMNS)
            wickets_col = _first_existing(purple_cap, _WICKETS_COLUMNS)
            if wickets_col and (team_col or player_col):
                team_wicket_totals: dict[str, dict[str, float]] = defaultdict(dict)
                for index, row in purple_cap.iterrows():
                    team = self._resolve_team_from_caps_row(row, team_col, player_col, player_teams)
                    wickets = _coerce_float(row.get(wickets_col))
                    if not team or wickets is None:
                        continue
                    player = _player_key(row.get(player_col)) if player_col else str(index)
                    team_wicket_totals[team][player or str(index)] = max(team_wicket_totals[team].get(player or str(index), 0.0), wickets)

                for team, values in team_wicket_totals.items():
                    current = leaders.get(team, TeamLeaderStats())
                    leaders[team] = TeamLeaderStats(
                        top_run_getters_runs=current.top_run_getters_runs,
                        top_wicket_takers_wickets=round(sum(sorted(values.values(), reverse=True)[:3]), 2),
                    )

        return leaders

    def _build_team_leader_names_from_caps(
        self,
        *,
        orange_cap: pd.DataFrame,
        purple_cap: pd.DataFrame,
        player_teams: dict[str, str],
    ) -> dict[str, TeamLeaderNames]:
        batter_names: dict[str, list[str]] = defaultdict(list)
        bowler_names: dict[str, list[str]] = defaultdict(list)

        if not orange_cap.empty:
            team_col = _first_existing(orange_cap, _TEAM_COLUMNS + _BATTING_TEAM_COLUMNS)
            player_col = _first_existing(orange_cap, _PLAYER_COLUMNS)
            runs_col = _first_existing(orange_cap, _RUNS_COLUMNS)
            if player_col and runs_col:
                top_batters = self._build_ranked_cap_entries(
                    frame=orange_cap,
                    team_col=team_col,
                    player_col=player_col,
                    value_col=runs_col,
                    player_teams=player_teams,
                )
                for team, values in top_batters.items():
                    batter_names[team] = [name for name, _ in values[:3]]

        if not purple_cap.empty:
            team_col = _first_existing(purple_cap, _TEAM_COLUMNS + _BOWLING_TEAM_COLUMNS)
            player_col = _first_existing(purple_cap, _PLAYER_COLUMNS)
            wickets_col = _first_existing(purple_cap, _WICKETS_COLUMNS)
            if player_col and wickets_col:
                top_bowlers = self._build_ranked_cap_entries(
                    frame=purple_cap,
                    team_col=team_col,
                    player_col=player_col,
                    value_col=wickets_col,
                    player_teams=player_teams,
                )
                for team, values in top_bowlers.items():
                    bowler_names[team] = [name for name, _ in values[:3]]

        result: dict[str, TeamLeaderNames] = {}
        for team in set(batter_names) | set(bowler_names):
            result[team] = TeamLeaderNames(
                top_batters=tuple(batter_names.get(team, [])),
                top_bowlers=tuple(bowler_names.get(team, [])),
            )
        return result

    def _build_ranked_cap_entries(
        self,
        *,
        frame: pd.DataFrame,
        team_col: str | None,
        player_col: str,
        value_col: str,
        player_teams: dict[str, str],
    ) -> dict[str, list[tuple[str, float]]]:
        ranked: dict[str, dict[str, tuple[str, float]]] = defaultdict(dict)
        for _, row in frame.iterrows():
            team = self._resolve_team_from_caps_row(row, team_col, player_col, player_teams)
            player_name = _clean_text(row.get(player_col))
            player_key = _player_key(player_name)
            value = _coerce_float(row.get(value_col))
            if not team or not player_name or not player_key or value is None:
                continue

            current = ranked[team].get(player_key)
            if current is None or value > current[1]:
                ranked[team][player_key] = (player_name, value)

        return {
            team: sorted(entries.values(), key=lambda item: item[1], reverse=True)
            for team, entries in ranked.items()
        }

    def _build_player_team_lookup(self, squads: pd.DataFrame) -> dict[str, str]:
        if squads.empty:
            return {}

        team_col = _first_existing(squads, _TEAM_COLUMNS)
        player_col = _first_existing(squads, _PLAYER_COLUMNS)
        if not (team_col and player_col):
            return {}

        lookup: dict[str, str] = {}
        for _, row in squads.iterrows():
            player = _player_key(row.get(player_col))
            team = resolve_team_name(_clean_text(row.get(team_col)))
            if player and team:
                lookup[player] = team
        return lookup

    def _resolve_team_from_caps_row(
        self,
        row: pd.Series,
        team_col: str | None,
        player_col: str | None,
        player_teams: dict[str, str],
    ) -> str:
        team = resolve_team_name(_clean_text(row.get(team_col))) if team_col else ""
        if team:
            return team
        if not player_col:
            return ""
        return player_teams.get(_player_key(row.get(player_col)), "")

    def _merge_team_leader_stats(
        self,
        primary: dict[str, TeamLeaderStats],
        fallback: dict[str, TeamLeaderStats],
    ) -> dict[str, TeamLeaderStats]:
        if not primary:
            return fallback
        if not fallback:
            return primary

        merged: dict[str, TeamLeaderStats] = {}
        for team in set(primary) | set(fallback):
            primary_stats = primary.get(team, TeamLeaderStats())
            fallback_stats = fallback.get(team, TeamLeaderStats())
            merged[team] = TeamLeaderStats(
                top_run_getters_runs=primary_stats.top_run_getters_runs
                if primary_stats.top_run_getters_runs > 0
                else fallback_stats.top_run_getters_runs,
                top_wicket_takers_wickets=primary_stats.top_wicket_takers_wickets
                if primary_stats.top_wicket_takers_wickets > 0
                else fallback_stats.top_wicket_takers_wickets,
            )
        return merged

    def _build_head_to_head_lookup(self, matches: pd.DataFrame) -> dict[frozenset[str], list[tuple[str, str]]]:
        team_a_col = _first_existing(matches, _TEAM_A_COLUMNS)
        team_b_col = _first_existing(matches, _TEAM_B_COLUMNS)
        winner_col = _first_existing(matches, _WINNER_COLUMNS)
        date_col = _first_existing(matches, _MATCH_DATE_COLUMNS)
        if not (team_a_col and team_b_col and winner_col and date_col):
            return {}

        lookup: dict[frozenset[str], list[tuple[str, str]]] = {}
        for _, row in matches.iterrows():
            team_a = resolve_team_name(_clean_text(row.get(team_a_col)))
            team_b = resolve_team_name(_clean_text(row.get(team_b_col)))
            winner = resolve_team_name(_clean_text(row.get(winner_col)))
            match_date = _parse_date(row.get(date_col))
            if not (team_a and team_b and winner and match_date):
                continue
            key = frozenset((team_a, team_b))
            lookup.setdefault(key, []).append((match_date, winner))

        for key, values in lookup.items():
            lookup[key] = sorted(values, key=lambda item: item[0])
        return lookup

    def _head_to_head_pct(
        self,
        lookup: dict[frozenset[str], list[tuple[str, str]]],
        team_a: str,
        team_b: str,
        limit: int = 7,
    ) -> float:
        results = lookup.get(frozenset((team_a, team_b)))
        if not results:
            return 0.5
        recent = results[-limit:]
        if not recent:
            return 0.5
        wins = sum(1 for _, winner in recent if winner == team_a)
        return round(wins / len(recent), 3)

    def _is_pending_match(self, match_date: str, winner: str, status: str, today: str) -> bool:
        if winner:
            return False
        if any(keyword in status for keyword in ("live", "upcoming", "scheduled", "preview", "toss")):
            return True
        if status and any(keyword in status for keyword in ("completed", "result", "won by", "abandoned")):
            return False
        return bool(match_date) and match_date >= today