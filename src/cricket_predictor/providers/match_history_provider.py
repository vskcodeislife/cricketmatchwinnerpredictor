from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from cricket_predictor.providers.cricinfo_standings import resolve_team_name


@dataclass(frozen=True)
class HistoricalMatchResult:
    match_date: str
    team_a: str
    team_b: str
    winner: str


class MatchHistoryProvider:
    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)

    def recent_form(self, team: str, limit: int = 5, fallback: float = 0.5) -> float:
        canonical = resolve_team_name(team)
        recent: list[int] = []
        for result in reversed(self._load_results()):
            if canonical not in {result.team_a, result.team_b}:
                continue
            recent.append(int(result.winner == canonical))
            if len(recent) == limit:
                break
        if not recent:
            return fallback
        return round(sum(recent) / len(recent), 3)

    def head_to_head_pct(
        self,
        team_a: str,
        team_b: str,
        limit: int = 7,
        fallback: float = 0.5,
    ) -> float:
        canonical_a = resolve_team_name(team_a)
        canonical_b = resolve_team_name(team_b)
        recent: list[int] = []
        for result in reversed(self._load_results()):
            if {canonical_a, canonical_b} != {result.team_a, result.team_b}:
                continue
            recent.append(int(result.winner == canonical_a))
            if len(recent) == limit:
                break
        if not recent:
            return fallback
        return round(sum(recent) / len(recent), 3)

    def _load_results(self) -> list[HistoricalMatchResult]:
        return _load_results_from_root(str(self._data_dir.resolve()))


@lru_cache(maxsize=4)
def _load_results_from_root(root_dir: str) -> list[HistoricalMatchResult]:
    root = Path(root_dir)
    results: dict[tuple[str, str, str], HistoricalMatchResult] = {}
    for folder_name in ("ipl_male_json", "recently_played_30_male_json"):
        folder = root / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                info = data.get("info", {})
                teams = info.get("teams", [])
                dates = info.get("dates", [])
                outcome = info.get("outcome", {})
                winner = resolve_team_name(str(outcome.get("winner") or "").strip())
                if len(teams) != 2 or not dates or not winner:
                    continue
                if outcome.get("result") in {"no result", "tie"}:
                    continue
                team_a = resolve_team_name(str(teams[0]).strip())
                team_b = resolve_team_name(str(teams[1]).strip())
                match_date = str(dates[0])
                key = (*sorted((team_a, team_b)), match_date)
                results[key] = HistoricalMatchResult(
                    match_date=match_date,
                    team_a=team_a,
                    team_b=team_b,
                    winner=winner,
                )
            except Exception:  # noqa: BLE001
                continue
    return sorted(results.values(), key=lambda result: result.match_date)