"""Fetch live IPL data from the iplt20.com S3 JSONP feeds.

The official IPL stats site uses JSONP endpoints backed by S3:
  - ``284-toprunsscorers.js``  → Orange Cap (most runs)
  - ``284-mostwickets.js``     → Purple Cap (most wickets)
  - ``284-groupstandings.js``  → Points table / group standings

``284`` is the IPL 2026 competition ID.  The data is refreshed after every
match by Sports Mechanic and requires no authentication — only a ``Referer``
header pointing to ``https://www.iplt20.com/``.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

import httpx
import truststore

from cricket_predictor.providers.cricinfo_standings import (
    RecentMatchResult,
    TeamStanding,
    resolve_team_name,
    short_code,
)
from cricket_predictor.providers.ipl_csv_provider import TeamLeaderStats

log = logging.getLogger(__name__)

# Inject OS-level trust store so S3 certs validate on all platforms
truststore.inject_into_ssl()

_BASE_URL = "https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/feeds/stats"
_HEADERS = {
    "Referer": "https://www.iplt20.com/",
    "User-Agent": "Mozilla/5.0 CricketPredictor/1.0",
}
_TIMEOUT = 10


def _parse_jsonp(text: str) -> dict:
    """Strip the JSONP callback wrapper and return the inner JSON object."""
    return json.loads(re.sub(r"^[^(]+\(|\);?\s*$", "", text))


def _fetch_jsonp(path: str) -> dict | None:
    """Fetch a JSONP feed from the S3 bucket and return parsed JSON."""
    url = f"{_BASE_URL}/{path}"
    try:
        resp = httpx.get(url, headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        return _parse_jsonp(resp.text)
    except Exception as exc:
        log.warning("iplt20 stats fetch failed (%s): %s", path, exc)
        return None


def fetch_team_leader_stats(competition_id: str = "284") -> dict[str, TeamLeaderStats] | None:
    """Return per-team leader stats from iplt20.com S3 feeds.

    Returns ``None`` when both feeds fail so the caller can fall back to
    another source.
    """
    runs_data = _fetch_jsonp(f"{competition_id}-toprunsscorers.js")
    wickets_data = _fetch_jsonp(f"{competition_id}-mostwickets.js")

    if runs_data is None and wickets_data is None:
        return None

    # Aggregate: sum runs for top 3 batters per team, sum wickets for top 3 bowlers
    team_runs: dict[str, float] = defaultdict(float)
    team_wickets: dict[str, float] = defaultdict(float)
    team_run_count: dict[str, int] = defaultdict(int)
    team_wicket_count: dict[str, int] = defaultdict(int)

    if runs_data:
        for player in runs_data.get("toprunsscorers", []):
            team = resolve_team_name(player.get("TeamName", ""))
            if not team or team_run_count[team] >= 3:
                continue
            team_run_count[team] += 1
            team_runs[team] += float(player.get("TotalRuns", 0))

    if wickets_data:
        for player in wickets_data.get("mostwickets", []):
            team = resolve_team_name(player.get("TeamName", ""))
            if not team or team_wicket_count[team] >= 3:
                continue
            team_wicket_count[team] += 1
            team_wickets[team] += float(player.get("Wickets", 0))

    all_teams = set(team_runs) | set(team_wickets)
    if not all_teams:
        return None

    result: dict[str, TeamLeaderStats] = {}
    for team in all_teams:
        result[team] = TeamLeaderStats(
            top_run_getters_runs=team_runs.get(team, 0.0),
            top_wicket_takers_wickets=team_wickets.get(team, 0.0),
        )

    log.info(
        "iplt20 stats: loaded leaders for %d teams (runs=%d entries, wickets=%d entries).",
        len(result),
        sum(team_run_count.values()),
        sum(team_wicket_count.values()),
    )
    return result


# ---------------------------------------------------------------------------
# Standings (Points Table)
# ---------------------------------------------------------------------------

def _derive_strength(total_runs: float, overs: float, matches: int, *, is_batting: bool) -> float:
    """Convert total runs / overs into a 40–90 strength score.

    Uses the same T20 calibration as ``cricinfo_standings.py``:
      batting:  avg runs per innings 100→40, 230→90
      bowling:  avg runs conceded per innings 100→90 (good), 230→40 (bad)
    """
    if matches <= 0 or overs <= 0:
        return 65.0
    avg_per_innings = total_runs / matches
    avg_per_innings = max(100.0, min(230.0, avg_per_innings))
    if is_batting:
        return round(40.0 + (avg_per_innings - 100.0) / 130.0 * 50.0, 2)
    return round(40.0 + (230.0 - avg_per_innings) / 130.0 * 50.0, 2)


def _parse_runs_overs(value: str) -> tuple[float, float]:
    """Parse ``'849/81.1'`` into ``(849.0, 81.1)``."""
    parts = value.split("/")
    if len(parts) != 2:
        return 0.0, 0.0
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return 0.0, 0.0


def _form_pct(performance: str) -> float:
    """Convert ``'W,W,L,W'`` to a win-rate float."""
    results = [r.strip() for r in performance.split(",") if r.strip()]
    if not results:
        return 0.5
    wins = sum(1 for r in results if r == "W")
    decided = sum(1 for r in results if r in ("W", "L"))
    return wins / decided if decided else 0.5


def fetch_standings(
    competition_id: str = "284",
) -> list[TeamStanding] | None:
    """Return standings from the iplt20.com S3 feed, or ``None`` on failure."""
    data = _fetch_jsonp(f"{competition_id}-groupstandings.js")
    if data is None:
        return None

    entries = data.get("points", [])
    if not entries:
        return None

    standings: list[TeamStanding] = []
    for entry in entries:
        team = resolve_team_name(entry.get("TeamName", ""))
        if not team:
            continue

        matches = int(entry.get("Matches", 0))
        for_runs, for_overs = _parse_runs_overs(entry.get("ForTeams", "0/0"))
        against_runs, against_overs = _parse_runs_overs(entry.get("AgainstTeam", "0/0"))
        performance = entry.get("Performance", "")

        standings.append(
            TeamStanding(
                team=team,
                short=short_code(team),
                position=int(entry.get("OrderNo", 0)),
                played=matches,
                won=int(entry.get("Wins", 0)),
                lost=int(entry.get("Loss", 0)),
                tied=int(entry.get("Tied", 0)),
                no_result=int(entry.get("NoResult", 0)),
                points=int(entry.get("Points", 0)),
                nrr=float(entry.get("NetRunRate", 0)),
                recent_form_str=performance.replace(",", " "),
                recent_form_pct=_form_pct(performance),
                batting_strength=_derive_strength(for_runs, for_overs, matches, is_batting=True),
                bowling_strength=_derive_strength(against_runs, against_overs, matches, is_batting=False),
            )
        )

    standings.sort(key=lambda s: s.position)
    log.info("iplt20 standings: loaded %d teams.", len(standings))
    return standings
