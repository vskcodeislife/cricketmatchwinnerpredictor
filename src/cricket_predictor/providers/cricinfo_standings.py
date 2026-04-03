"""IPL standings provider — parses the Delhi Capitals points-table page.

The page embeds a ``globalThis.standings_*`` SSR JSON object that contains
the live IPL standings from Cricbuzz. No JavaScript execution is required —
the full table is baked into the initial HTML response.

Data shape (per team in ``standingsData[0].teams``):
    pos, name, short_name, id, p, w, l, t, nr, pts, nrr
    matches[]:  list of recent match results including score arrays.

Recent form (W/L) is derived from the ``matches`` array: for each match,
when the team_id of the winner's innings (lower wickets-lost or chase success)
matches this team's ``id``, record a "W", else "L".
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Team name → canonical long name  (covers short codes AND common variants)
# ---------------------------------------------------------------------------
_ALIAS_MAP: dict[str, str] = {
    "rr": "Rajasthan Royals",
    "rajasthan": "Rajasthan Royals",
    "rajasthan royals": "Rajasthan Royals",
    "rcb": "Royal Challengers Bengaluru",
    "royal challengers bengaluru": "Royal Challengers Bengaluru",
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "bengaluru": "Royal Challengers Bengaluru",
    "bangalore": "Royal Challengers Bengaluru",
    "dc": "Delhi Capitals",
    "delhi": "Delhi Capitals",
    "delhi capitals": "Delhi Capitals",
    "mi": "Mumbai Indians",
    "mumbai": "Mumbai Indians",
    "mumbai indians": "Mumbai Indians",
    "pbks": "Punjab Kings",
    "punjab": "Punjab Kings",
    "punjab kings": "Punjab Kings",
    "gt": "Gujarat Titans",
    "gujarat": "Gujarat Titans",
    "gujarat titans": "Gujarat Titans",
    "kkr": "Kolkata Knight Riders",
    "kolkata": "Kolkata Knight Riders",
    "kolkata knight riders": "Kolkata Knight Riders",
    "lsg": "Lucknow Super Giants",
    "lucknow": "Lucknow Super Giants",
    "lucknow super giants": "Lucknow Super Giants",
    "srh": "Sunrisers Hyderabad",
    "hyderabad": "Sunrisers Hyderabad",
    "sunrisers hyderabad": "Sunrisers Hyderabad",
    "sunrisers": "Sunrisers Hyderabad",
    "csk": "Chennai Super Kings",
    "chennai": "Chennai Super Kings",
    "chennai super kings": "Chennai Super Kings",
}

_SHORT_CODE: dict[str, str] = {
    "Rajasthan Royals": "RR",
    "Royal Challengers Bengaluru": "RCB",
    "Delhi Capitals": "DC",
    "Mumbai Indians": "MI",
    "Punjab Kings": "PBKS",
    "Gujarat Titans": "GT",
    "Kolkata Knight Riders": "KKR",
    "Lucknow Super Giants": "LSG",
    "Sunrisers Hyderabad": "SRH",
    "Chennai Super Kings": "CSK",
}

VENUE_HOME_MAP: dict[str, str] = {
    "rajiv gandhi international stadium": "Sunrisers Hyderabad",
    "eden gardens": "Kolkata Knight Riders",
    "wankhede stadium": "Mumbai Indians",
    "ma chidambaram stadium": "Chennai Super Kings",
    "chepauk": "Chennai Super Kings",
    "m chinnaswamy stadium": "Royal Challengers Bengaluru",
    "chinnaswamy": "Royal Challengers Bengaluru",
    "narendra modi stadium": "Gujarat Titans",
    "arun jaitley stadium": "Delhi Capitals",
    "sawai mansingh stadium": "Rajasthan Royals",
    "aca stadium, barsapara": "Rajasthan Royals",
    "barsapara": "Rajasthan Royals",
    "bharat ratna shri atal bihari vajpayee ekana cricket stadium": "Lucknow Super Giants",
    "ekana": "Lucknow Super Giants",
    "punjab cricket association is bindra stadium": "Punjab Kings",
    "bindra stadium": "Punjab Kings",
    "new international cricket stadium": "Punjab Kings",
    "himachal pradesh cricket association stadium": "Punjab Kings",
}


def resolve_team_name(raw: str) -> str:
    """Return canonical team name for any recognised alias or short code."""
    return _ALIAS_MAP.get(raw.strip().lower(), raw.strip())


def short_code(canonical: str) -> str:
    return _SHORT_CODE.get(canonical, canonical[:3].upper())


def venue_advantage(venue: str, team_a: str, team_b: str) -> float:
    lower = venue.lower()
    for keyword, home_team in VENUE_HOME_MAP.items():
        if keyword in lower:
            if home_team == team_a:
                return 1.0
            if home_team == team_b:
                return -1.0
            return 0.0
    return 0.0


@dataclass
class TeamStanding:
    team: str
    short: str
    position: int
    played: int
    won: int
    lost: int
    tied: int
    no_result: int
    points: int
    nrr: float
    recent_form_str: str
    recent_form_pct: float
    batting_strength: float
    bowling_strength: float
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))


class CricinfoStandingsProvider:
    """Fetches IPL standings from the Delhi Capitals points-table page.

    The page embeds live Cricbuzz standings as a server-side rendered JSON
    object (``globalThis.standings_*``). We extract it with brace-counting
    rather than a brittle regex so it handles any level of nesting.
    """

    def __init__(self, page_url: str, timeout_seconds: float = 15.0) -> None:
        self._url = page_url
        self._timeout = timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self) -> list[TeamStanding]:
        try:
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
        import httpx

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/",
        }
        resp = httpx.get(self._url, headers=headers, timeout=self._timeout, follow_redirects=True)
        resp.raise_for_status()
        return self._parse(resp.text)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _extract_ssr_json(self, html: str) -> dict[str, Any]:
        """Extract the first globalThis.standings_* JSON object via brace counting."""
        start_match = re.search(r'globalThis\.standings_\w+\s*=\s*\{', html)
        if not start_match:
            raise ValueError("globalThis.standings_* not found in page HTML.")
        start = start_match.end() - 1
        depth, end = 0, start
        for i, ch in enumerate(html[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        return json.loads(html[start:end])

    def _derive_form(self, team_id: str, matches: list[dict[str, Any]]) -> str:
        """Return a space-separated form string e.g. 'W L W W' from recent matches."""
        results: list[str] = []
        for match in matches:
            scores = match.get("score", [])
            if len(scores) < 2:
                continue
            # Determine winner: in a T20 chase the 2nd innings team wins if they
            # reach the target. Proxy: team with fewer wickets lost (lower overs)
            # and successfully chasing. Simplest reliable heuristic: compare runs.
            team_runs: dict[str, int] = {}
            for s in scores:
                tid = str(s.get("team_id", ""))
                innings = s.get("innings", [])
                runs = sum(int(inn.get("runs_scored", 0)) for inn in innings)
                team_runs[tid] = team_runs.get(tid, 0) + runs

            if not team_runs:
                continue
            winner_id = max(team_runs, key=lambda k: team_runs[k])
            results.append("W" if winner_id == team_id else "L")

        return " ".join(results[-5:]) if results else ""

    def _derive_batting_bowling(
        self, team_id: str, matches: list[dict[str, Any]]
    ) -> tuple[float, float]:
        """Return (batting_strength, bowling_strength) derived from match scores.

        batting_strength  = avg runs scored per innings, scaled to 40–90.
        bowling_strength  = avg runs conceded per innings, inverted and scaled
                            to 40–90 (lower conceded → higher bowling strength).

        T20 calibration: 120 → 40, 220 → 90 for batting;
                         120 → 90, 220 → 40 for bowling (fewer conceded = better).
        Returns (65.0, 65.0) when no score data is available so the model
        gets a neutral prior rather than noise.
        """
        scored: list[int] = []
        conceded: list[int] = []

        for match in matches:
            runs_by_team: dict[str, int] = {}
            for s in match.get("score", []):
                tid = str(s.get("team_id", ""))
                innings = s.get("innings", [])
                runs = sum(int(inn.get("runs_scored", 0)) for inn in innings)
                runs_by_team[tid] = runs_by_team.get(tid, 0) + runs

            if team_id in runs_by_team:
                scored.append(runs_by_team[team_id])
                for tid, runs in runs_by_team.items():
                    if tid != team_id:
                        conceded.append(runs)

        if not scored:
            return 65.0, 65.0

        avg_scored   = sum(scored)   / len(scored)
        avg_conceded = sum(conceded) / len(conceded) if conceded else avg_scored

        # Clamp to realistic T20 range 100–230 before scaling
        avg_scored   = max(100.0, min(230.0, avg_scored))
        avg_conceded = max(100.0, min(230.0, avg_conceded))

        bat  = round(40.0 + (avg_scored   - 100.0) / 130.0 * 50.0, 2)
        bowl = round(40.0 + (230.0 - avg_conceded) / 130.0 * 50.0, 2)
        return bat, bowl

    def _parse(self, html: str) -> list[TeamStanding]:
        data = self._extract_ssr_json(html)
        raw_teams: list[dict[str, Any]] = data.get("standingsData", [{}])[0].get("teams", [])

        if not raw_teams:
            log.warning("No teams found in standings JSON. Table may not be available yet.")
            return []

        standings: list[TeamStanding] = []
        now = datetime.utcnow().isoformat(timespec="seconds")

        for t in raw_teams:
            raw_name   = str(t.get("name", "")).strip()
            canonical  = resolve_team_name(raw_name) or raw_name
            team_id    = str(t.get("id", ""))
            played     = int(t.get("p", 0))
            won        = int(t.get("w", 0))
            lost       = int(t.get("l", 0))
            tied       = int(t.get("t", 0))
            no_result  = int(t.get("nr", 0))
            points     = int(t.get("pts", 0))
            nrr        = float(str(t.get("nrr", "0")).replace("+", "") or 0)
            position   = int(t.get("pos", 99))

            # Recent form from the embedded match results
            recent_str = self._derive_form(team_id, t.get("matches", []))

            # recent_form_pct: wins / played; fall back to W/P ratio
            recent_form_pct = (won / played) if played > 0 else 0.5

            # Batting/bowling strength from actual runs scored/conceded.
            # This separates the two dimensions correctly — NRR conflates them
            # (e.g. SRH scored 201 but conceded quickly → bad NRR despite good
            # batting) so using NRR as a proxy for bowling produces wrong signals.
            batting_strength, bowling_strength = self._derive_batting_bowling(
                team_id, t.get("matches", [])
            )

            standings.append(
                TeamStanding(
                    team=canonical,
                    short=t.get("short_name", short_code(canonical)),
                    position=position,
                    played=played,
                    won=won,
                    lost=lost,
                    tied=tied,
                    no_result=no_result,
                    points=points,
                    nrr=nrr,
                    recent_form_str=recent_str,
                    recent_form_pct=round(recent_form_pct, 3),
                    batting_strength=batting_strength,
                    bowling_strength=bowling_strength,
                    fetched_at=now,
                )
            )

        standings.sort(key=lambda s: s.position)
        log.info("Parsed %d team standings from %s", len(standings), self._url)
        return standings


