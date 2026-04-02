"""Standings service — caches the latest IPL points table in memory.

A background task in ``app.py`` calls ``refresh()`` periodically.
All prediction code calls ``get()`` which returns the cached snapshot.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from cricket_predictor.config.settings import Settings, get_settings
from cricket_predictor.providers.cricinfo_standings import (
    CricinfoStandingsProvider,
    TeamStanding,
    resolve_team_name,
)

log = logging.getLogger(__name__)


class StandingsService:
    def __init__(self, settings: Settings) -> None:
        self._provider = CricinfoStandingsProvider(settings.cricinfo_standings_url)
        self._cache: dict[str, TeamStanding] = {}
        self._fetched_at: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def refresh(self) -> dict[str, TeamStanding]:
        """Fetch fresh standings in a thread pool and update the cache."""
        standings: list[TeamStanding] = await asyncio.to_thread(self._provider.fetch)
        self._cache = {s.team: s for s in standings}
        if standings:
            self._fetched_at = standings[0].fetched_at
            log.info("Standings refreshed for %d teams.", len(standings))
        return self._cache

    def get(self) -> dict[str, TeamStanding]:
        """Return the in-memory cache (empty dict if never refreshed)."""
        return dict(self._cache)

    def get_team(self, raw_name: str) -> TeamStanding | None:
        """Look up a team by any name alias. Returns None if not found."""
        canonical = resolve_team_name(raw_name)
        return self._cache.get(canonical)

    def recent_form(self, raw_name: str, fallback: float = 0.5) -> float:
        """Return win-rate (0–1) for a team, or fallback if not in cache."""
        standing = self.get_team(raw_name)
        return standing.recent_form_pct if standing is not None else fallback

    def batting_strength(self, raw_name: str, fallback: float = 65.0) -> float:
        standing = self.get_team(raw_name)
        return standing.batting_strength if standing is not None else fallback

    def bowling_strength(self, raw_name: str, fallback: float = 65.0) -> float:
        standing = self.get_team(raw_name)
        return standing.bowling_strength if standing is not None else fallback

    def as_table(self) -> list[dict]:
        """Return a list of dicts sorted by position for API responses."""
        rows = sorted(self._cache.values(), key=lambda s: s.position)
        return [
            {
                "position": s.position,
                "team": s.team,
                "short": s.short,
                "played": s.played,
                "won": s.won,
                "lost": s.lost,
                "tied": s.tied,
                "no_result": s.no_result,
                "points": s.points,
                "nrr": s.nrr,
                "recent_form": s.recent_form_str,
                "recent_form_pct": s.recent_form_pct,
            }
            for s in rows
        ]

    @property
    def fetched_at(self) -> str:
        return self._fetched_at


@lru_cache
def get_standings_service() -> StandingsService:
    return StandingsService(get_settings())
