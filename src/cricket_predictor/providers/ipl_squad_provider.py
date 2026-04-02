"""IPL squad provider — scrapes current squad from iplt20.com.

Each team page at https://www.iplt20.com/teams/{slug} embeds the full squad
in server-side rendered HTML.  Player anchors follow the pattern:
  <a href="/players/{player-slug}/{player-id}">NAME ROLE</a>

The squad is cached to ``data/squad_profiles.json`` so downstream code can
read it without network I/O.  Re-run ``scripts/fetch_squads.py`` any time
you want a fresh pull (e.g., after auction / mid-season trades).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Team slugs — matches the iplt20.com URL path
# ---------------------------------------------------------------------------
TEAM_SLUGS: dict[str, str] = {
    "Chennai Super Kings":        "chennai-super-kings",
    "Delhi Capitals":             "delhi-capitals",
    "Gujarat Titans":             "gujarat-titans",
    "Kolkata Knight Riders":      "kolkata-knight-riders",
    "Lucknow Super Giants":       "lucknow-super-giants",
    "Mumbai Indians":             "mumbai-indians",
    "Punjab Kings":               "punjab-kings",
    "Rajasthan Royals":           "rajasthan-royals",
    "Royal Challengers Bengaluru":"royal-challengers-bengaluru",
    "Sunrisers Hyderabad":        "sunrisers-hyderabad",
}

# Canonical roles derived from text following the player name on the page
_ROLE_TOKENS: list[str] = [
    "WK-Batter",
    "All-Rounder",
    "Batter",
    "Bowler",
]

_BASE_URL = "https://www.iplt20.com/teams"
_DEFAULT_CACHE = Path(__file__).parents[3] / "data" / "squad_profiles.json"


def _extract_role(text: str) -> tuple[str, str]:
    """Split 'TRAVIS HEAD Batter' → ('Travis Head', 'Batter')."""
    text = text.strip()
    for role in _ROLE_TOKENS:
        if text.upper().endswith(role.upper()):
            name = text[: -len(role)].strip().title()
            return name, role
    # Fallback: last word as role
    parts = text.rsplit(None, 1)
    if len(parts) == 2:
        return parts[0].strip().title(), parts[1].strip().title()
    return text.title(), "Unknown"


def _parse_squad_html(html: str) -> list[dict[str, str]]:
    """Extract all player entries from a team HTML page.

    The page uses Angular (ng-app) with SSR.  Each player card is an anchor:
      <a data-player_name="Ajinkya Rahane"
         href="https://www.iplt20.com/players/ajinkya-rahane/135 ">
        ...
        <h2>Ajinkya Rahane</h2>
        <span class="d-block w-100 text-center">Batter</span>
        ...
      </a>
    """
    players: list[dict[str, str]] = []
    seen: set[str] = set()

    # Each player card: grab the anchor with data-player_name + href=/players/...
    card_pattern = re.compile(
        r'data-player_name=["\']([^"\']+)["\'][^>]*'
        r'href=["\']https?://www\.iplt20\.com/players/([^/]+)/(\d+)\s*["\']'
        r'.*?'                          # content between opening anchor and role span
        r'class=["\'][^"\']*d-block[^"\']*text-center[^"\']*["\'][^>]*>([^<]+)<',
        re.DOTALL,
    )

    for m in card_pattern.finditer(html):
        name, slug, player_id, role = (
            m.group(1).strip(),
            m.group(2).strip(),
            m.group(3).strip(),
            m.group(4).strip(),
        )
        if player_id in seen or not name:
            continue
        seen.add(player_id)
        players.append({"name": name, "role": role, "player_id": player_id, "slug": slug})

    return players


class IPLSquadProvider:
    """Fetches and caches IPL 2026 squad data from iplt20.com."""

    def __init__(
        self,
        cache_path: str | Path = _DEFAULT_CACHE,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._cache_path = Path(cache_path)
        self._timeout = timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self) -> dict[str, Any]:
        """Fetch squads for all 10 teams and return the combined dict."""
        import httpx
        try:
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        squads: dict[str, Any] = {}
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with httpx.Client(headers=headers, timeout=self._timeout, follow_redirects=True) as client:
            for team, slug in TEAM_SLUGS.items():
                url = f"{_BASE_URL}/{slug}"
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    players = _parse_squad_html(resp.text)
                    squads[team] = {
                        "players": players,
                        "batters":      [p for p in players if "Batter" in p["role"]],
                        "all_rounders": [p for p in players if "All-Rounder" in p["role"]],
                        "bowlers":      [p for p in players if p["role"] == "Bowler"],
                        "total":        len(players),
                        "fetched_at":   now,
                    }
                    log.info("%-30s  %d players", team, len(players))
                except Exception as exc:
                    log.warning("Failed to fetch squad for %s: %s", team, exc)
                    squads[team] = {"players": [], "total": 0, "fetched_at": now, "error": str(exc)}

        return squads

    def save(self, squads: dict[str, Any] | None = None) -> Path:
        """Fetch (if not provided) and write cache file."""
        if squads is None:
            squads = self.fetch_all()
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(squads, indent=2))
        log.info("Squad cache written to %s", self._cache_path)
        return self._cache_path

    def load(self) -> dict[str, Any]:
        """Return cached squads, or empty dict if cache does not exist."""
        if not self._cache_path.exists():
            return {}
        return json.loads(self._cache_path.read_text())

    def get_team_squad(self, team: str) -> list[dict[str, str]]:
        """Return the player list for a team (from cache)."""
        return self.load().get(team, {}).get("players", [])

    def bowling_strength_score(self, team: str) -> float:
        """Return a bowling-quality score 0-1 based on squad composition.

        Counts specialist bowlers + all-rounders with a bowling bias and
        normalises to [0, 1].  Higher = stronger attack.
        """
        data = self.load().get(team, {})
        bowlers      = len(data.get("bowlers", []))
        all_rounders = len(data.get("all_rounders", []))
        # Weight: each specialist bowler ≈ 2× an all-rounder
        score = bowlers * 2 + all_rounders
        # A squad typically has 6-8 bowlers + 4-6 all-rounders → max ~22
        return round(min(1.0, score / 22.0), 3)

    def batting_strength_score(self, team: str) -> float:
        """Return a batting-quality score 0-1 based on squad composition."""
        data = self.load().get(team, {})
        batters      = len(data.get("batters", []))
        all_rounders = len(data.get("all_rounders", []))
        score = batters * 2 + all_rounders
        return round(min(1.0, score / 22.0), 3)
