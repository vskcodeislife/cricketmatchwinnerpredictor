"""Fetch live Orange Cap / Purple Cap data from the iplt20.com S3 JSONP feeds.

The official IPL stats site uses JSONP endpoints backed by S3:
  - ``284-toprunsscorers.js``  → Orange Cap (most runs)
  - ``284-mostwickets.js``     → Purple Cap (most wickets)

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

from cricket_predictor.providers.cricinfo_standings import resolve_team_name
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
