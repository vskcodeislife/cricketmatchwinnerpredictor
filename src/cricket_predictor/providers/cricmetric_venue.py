"""Scrape venue behavioral stats from cricmetric.com and cache locally.

The provider fetches stadium-level T20I data for every IPL venue and exposes it
as a ``VenueProfile`` dict suitable for use in the feature pipeline.

Parsed fields
-------------
avg_first_innings_score
    Most-recent year's average first-innings score in T20Is at this ground.
chase_win_pct
    Fraction of T20I matches won by the chasing side.
spin_economy
    Mean economy rate of spin bowlers (Orthodox, Legbreak, Offbreak, Chinaman).
pace_economy
    Mean economy rate of pace bowlers (Fast, Medium).
spin_wicket_pct
    Estimated fraction of wickets taken by spinners.
pace_wicket_pct
    Estimated fraction of wickets taken by pace bowlers.
boundary_rate
    Placeholder – derived as (avg_first_innings_score / 120) − 1, clipped to [0.2, 0.4].
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TypedDict

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class VenueProfile(TypedDict):
    avg_first_innings_score: float
    chase_win_pct: float
    spin_wicket_pct: float
    pace_wicket_pct: float
    boundary_rate: float
    spin_economy: float
    pace_economy: float


# ---------------------------------------------------------------------------
# IPL venue → cricmetric query string mapping
# ---------------------------------------------------------------------------

# Each entry maps the canonical venue name used in this codebase to the
# query string accepted by cricmetric (venue= URL parameter).
IPL_VENUE_MAP: dict[str, str] = {
    "Eden Gardens": "Eden+Gardens",
    "Wankhede Stadium": "Wankhede+Stadium",
    "MA Chidambaram Stadium": "Chennai",
    "M Chinnaswamy Stadium": "Chinnaswamy",
    "Arun Jaitley Stadium": "Arun+Jaitley+Stadium",
    "Rajiv Gandhi International Stadium": "Hyderabad+%28Deccan%29",
    "Narendra Modi Stadium": "Narendra+Modi+Stadium",
    "Sawai Mansingh Stadium": "Sawai+Mansingh+Stadium",
    "BRSABV Ekana Cricket Stadium": "Lucknow",
    "Himachal Pradesh Cricket Association Stadium": "Dharamsala",
    "ACA-VDCA Cricket Stadium": "Guwahati",
    "Raipur International Cricket Stadium": "Raipur",
    "Punjab Cricket Association IS Bindra Stadium": "Mohali",
    # Short city aliases also stored for direct lookup
    "Mumbai": "Wankhede+Stadium",
    "Kolkata": "Eden+Gardens",
    "Chennai": "Chennai",
    "Bangalore": "Chinnaswamy",
    "Bengaluru": "Chinnaswamy",
    "Delhi": "Arun+Jaitley+Stadium",
    "Hyderabad": "Hyderabad+%28Deccan%29",
    "Ahmedabad": "Narendra+Modi+Stadium",
    "Jaipur": "Sawai+Mansingh+Stadium",
    "Lucknow": "Lucknow",
    "Dharamshala": "Dharamsala",
    "Dharamsala": "Dharamsala",
    "Guwahati": "Guwahati",
    "Raipur": "Raipur",
    "Mohali": "Mohali",
    "Chandigarh": "Mohali",
}

_CRICMETRIC_BASE = "https://www.cricmetric.com/venue.py"
_DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "venue_profiles.json"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Referer": "https://www.cricmetric.com/",
}

# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

# Format: dataArray.push(['2024', 172.5000]);
_RE_YEAR_SCORE = re.compile(r"dataArray\.push\(\['(\d{4})',\s*([\d.]+)\]\)")

# Format: ['Team batting 2nd won', 7],['Team batting 1st won', 8]
_RE_RESULTS = re.compile(
    r"\['Team batting 2nd won',\s*(\d+)\][^[]*\['Team batting 1st won',\s*(\d+)\]",
    re.DOTALL,
)

# Format: [10.0122, 35.6957, 'Fast', 'Fast']
_RE_BOWLING = re.compile(r"\[([\d.]+),\s*([\d.]+),\s*'(\w+)',\s*'\w+'\]")

_SPIN_TYPES = {"Orthodox", "Legbreak", "Offbreak", "Chinaman"}
_PACE_TYPES = {"Fast", "Medium"}

DEFAULT_PROFILE: VenueProfile = {
    "avg_first_innings_score": 165.0,
    "chase_win_pct": 0.50,
    "spin_wicket_pct": 0.40,
    "pace_wicket_pct": 0.60,
    "boundary_rate": 0.29,
    "spin_economy": 8.2,
    "pace_economy": 9.0,
}


def _parse_venue_html(html: str) -> VenueProfile | None:
    """Parse a cricmetric venue page and extract a :class:`VenueProfile`.

    Returns ``None`` if the page is a disambiguation page or lacks data.
    """
    # Guard: disambiguation pages don't contain the chart functions
    if "function drawChart_T20I()" not in html and "drawChart_T20I" not in html:
        logger.debug("Disambiguation or no-data page detected")
        return None

    profile: dict[str, float] = {}

    # --- Avg first-innings score: weighted mean across years (most recent year weighted 2x) ---
    year_scores = _RE_YEAR_SCORE.findall(html)
    if year_scores:
        valid = [(int(y), float(s)) for y, s in year_scores if float(s) > 0]
        if valid:
            # Weight most recent year 2x to reflect current conditions
            most_recent_year = max(y for y, _ in valid)
            weights = [2.0 if y == most_recent_year else 1.0 for y, _ in valid]
            weighted_sum = sum(s * w for (_, s), w in zip(valid, weights))
            total_weight = sum(weights)
            val = weighted_sum / total_weight
            # Cap at 220 to guard against single-match outliers exceeding T20 norms
            profile["avg_first_innings_score"] = round(min(val, 220.0), 2)

    # --- Results / chase-win % -----------------------------------------
    m = _RE_RESULTS.search(html)
    if m:
        chase_wins = int(m.group(1))
        first_wins = int(m.group(2))
        total = chase_wins + first_wins
        if total > 0:
            profile["chase_win_pct"] = round(chase_wins / total, 4)

    # --- Bowling by type ------------------------------------------------
    bowling_rows = _RE_BOWLING.findall(html)
    bowling: dict[str, dict[str, float]] = {}
    for economy_str, average_str, bowl_type in bowling_rows:
        economy = float(economy_str)
        average = float(average_str)
        if bowl_type not in bowling and average > 0:
            bowling[bowl_type] = {"economy": economy, "average": average}

    spin_entries = [(bowling[t]["economy"], bowling[t]["average"]) for t in _SPIN_TYPES if t in bowling]
    pace_entries = [(bowling[t]["economy"], bowling[t]["average"]) for t in _PACE_TYPES if t in bowling]

    if spin_entries:
        profile["spin_economy"] = round(sum(e for e, _ in spin_entries) / len(spin_entries), 4)
    if pace_entries:
        profile["pace_economy"] = round(sum(e for e, _ in pace_entries) / len(pace_entries), 4)

    # Estimate wicket shares using bowling_average (lower avg → more wickets per run)
    spin_wicket_rate = sum(1.0 / a for _, a in spin_entries) if spin_entries else 0.0
    pace_wicket_rate = sum(1.0 / a for _, a in pace_entries) if pace_entries else 0.0
    total_wkt_rate = spin_wicket_rate + pace_wicket_rate
    if total_wkt_rate > 0:
        profile["spin_wicket_pct"] = round(spin_wicket_rate / total_wkt_rate, 4)
        profile["pace_wicket_pct"] = round(pace_wicket_rate / total_wkt_rate, 4)

    # Boundary rate derived from scoring (approximation)
    if "avg_first_innings_score" in profile:
        score = profile["avg_first_innings_score"]
        # typical ball count 120; roughly 25-35% of balls go for boundaries in IPL
        raw = (score / 120.0) - 1.0
        profile["boundary_rate"] = round(max(0.20, min(0.40, raw + 0.28)), 4)

    if len(profile) < 3:
        return None

    result = DEFAULT_PROFILE.copy()
    result.update(profile)  # type: ignore[arg-type]
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class CricmetricVenueProvider:
    """Scrape and cache venue profiles from cricmetric.com."""

    def __init__(
        self,
        cache_path: Path | None = None,
        fetch_format: str = "T20I",
    ) -> None:
        self._cache_path = cache_path or _DEFAULT_CACHE_PATH
        self._format = fetch_format
        self._cache: dict[str, VenueProfile] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_cache(self) -> dict[str, VenueProfile]:
        """Load previously-fetched data from the JSON cache."""
        if self._cache_path.exists():
            try:
                raw = json.loads(self._cache_path.read_text())
                self._cache = {k: {**DEFAULT_PROFILE, **v} for k, v in raw.items()}
                logger.info("Loaded cricmetric venue cache: %d venues", len(self._cache))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load venue cache: %s", exc)
        return self._cache

    def save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache, indent=2))
        logger.info("Saved cricmetric venue cache to %s", self._cache_path)

    def fetch_venue(self, venue_name: str) -> VenueProfile | None:
        """Fetch a single venue by its canonical name or city alias."""
        query = IPL_VENUE_MAP.get(venue_name, venue_name.replace(" ", "+"))
        url = f"{_CRICMETRIC_BASE}?venue={query}&format={self._format}&category=Men"
        try:
            with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=20) as client:
                resp = client.get(url)
                resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch %s from cricmetric: %s", venue_name, exc)
            return None

        profile = _parse_venue_html(resp.text)
        if profile:
            self._cache[venue_name] = profile
        else:
            logger.warning("No parseable data for venue=%s (url=%s)", venue_name, url)
        return profile

    def fetch_all_ipl_venues(self, delay_secs: float = 1.0) -> dict[str, VenueProfile]:
        """Fetch all IPL venues from cricmetric and update the cache.

        Adds a small delay between requests to be a polite scraper.
        """
        # Deduplicate by canonical venue name (skip city-alias duplicates)
        canonical_venues = [v for v in IPL_VENUE_MAP if not _is_city_alias(v)]
        for venue_name in canonical_venues:
            logger.info("Fetching cricmetric data for: %s", venue_name)
            self.fetch_venue(venue_name)
            time.sleep(delay_secs)
        self.save_cache()
        return self._cache

    def get_profile(self, venue_name: str) -> VenueProfile:
        """Return a venue profile, using cache first, then fetch, then default."""
        if venue_name in self._cache:
            return self._cache[venue_name]
        # Try city aliases
        for alias, query in IPL_VENUE_MAP.items():
            if alias.lower() == venue_name.lower() and alias in self._cache:
                return self._cache[alias]
        return DEFAULT_PROFILE


def _is_city_alias(name: str) -> bool:
    """Return True if the name is a short city alias (single word or short name)."""
    return len(name.split()) <= 2 and name not in {
        "Eden Gardens", "Wankhede Stadium", "MA Chidambaram Stadium",
        "M Chinnaswamy Stadium", "Arun Jaitley Stadium",
        "Rajiv Gandhi International Stadium", "Narendra Modi Stadium",
        "Sawai Mansingh Stadium", "BRSABV Ekana Cricket Stadium",
        "Himachal Pradesh Cricket Association Stadium",
        "ACA-VDCA Cricket Stadium", "Raipur International Cricket Stadium",
        "Punjab Cricket Association IS Bindra Stadium",
    }
