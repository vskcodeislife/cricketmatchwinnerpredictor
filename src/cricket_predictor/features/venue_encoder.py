"""Venue behavioral profiles.

At runtime, the live cricmetric cache (``data/venue_profiles.json``) is merged
with the static fallback table so that every known venue always has a profile.

Features
--------
avg_first_innings_score
    Typical first-innings total in T20 matches at this venue.
chase_win_pct
    Fraction of T20 matches won by the chasing side historically.
spin_wicket_pct
    Fraction of wickets taken by spinners at this venue.
pace_wicket_pct
    Fraction of wickets taken by pace bowlers at this venue.
boundary_rate
    Fraction of legal deliveries that result in a boundary (4 or 6).
spin_economy
    Mean economy rate of spin bowlers (from cricmetric).
pace_economy
    Mean economy rate of pace bowlers (from cricmetric).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Venue profile lookup table
# ---------------------------------------------------------------------------
VENUE_PROFILES: dict[str, dict[str, float]] = {
    # ---- IPL / modern T20 venues ----
    "Wankhede Stadium": {
        "avg_first_innings_score": 175.0,
        "chase_win_pct": 0.48,
        "spin_wicket_pct": 0.38,
        "pace_wicket_pct": 0.62,
        "boundary_rate": 0.32,
    },
    "Eden Gardens": {
        "avg_first_innings_score": 168.0,
        "chase_win_pct": 0.50,
        "spin_wicket_pct": 0.42,
        "pace_wicket_pct": 0.58,
        "boundary_rate": 0.29,
    },
    "M Chinnaswamy Stadium": {
        "avg_first_innings_score": 185.0,
        "chase_win_pct": 0.45,
        "spin_wicket_pct": 0.40,
        "pace_wicket_pct": 0.60,
        "boundary_rate": 0.36,
    },
    "Arun Jaitley Stadium": {
        "avg_first_innings_score": 165.0,
        "chase_win_pct": 0.52,
        "spin_wicket_pct": 0.45,
        "pace_wicket_pct": 0.55,
        "boundary_rate": 0.28,
    },
    "MA Chidambaram Stadium": {
        "avg_first_innings_score": 158.0,
        "chase_win_pct": 0.55,
        "spin_wicket_pct": 0.55,
        "pace_wicket_pct": 0.45,
        "boundary_rate": 0.27,
    },
    "Rajiv Gandhi International Stadium": {
        "avg_first_innings_score": 172.0,
        "chase_win_pct": 0.53,
        "spin_wicket_pct": 0.43,
        "pace_wicket_pct": 0.57,
        "boundary_rate": 0.30,
    },
    "Narendra Modi Stadium": {
        "avg_first_innings_score": 170.0,
        "chase_win_pct": 0.51,
        "spin_wicket_pct": 0.44,
        "pace_wicket_pct": 0.56,
        "boundary_rate": 0.30,
    },
    "Punjab Cricket Association IS Bindra Stadium": {
        "avg_first_innings_score": 178.0,
        "chase_win_pct": 0.46,
        "spin_wicket_pct": 0.36,
        "pace_wicket_pct": 0.64,
        "boundary_rate": 0.33,
    },
    "Sawai Mansingh Stadium": {
        "avg_first_innings_score": 164.0,
        "chase_win_pct": 0.49,
        "spin_wicket_pct": 0.48,
        "pace_wicket_pct": 0.52,
        "boundary_rate": 0.28,
    },
    "DY Patil Stadium": {
        "avg_first_innings_score": 173.0,
        "chase_win_pct": 0.47,
        "spin_wicket_pct": 0.39,
        "pace_wicket_pct": 0.61,
        "boundary_rate": 0.31,
    },
    "Brabourne Stadium": {
        "avg_first_innings_score": 171.0,
        "chase_win_pct": 0.49,
        "spin_wicket_pct": 0.41,
        "pace_wicket_pct": 0.59,
        "boundary_rate": 0.30,
    },
    "BRSABV Ekana Cricket Stadium": {
        "avg_first_innings_score": 166.0,
        "chase_win_pct": 0.50,
        "spin_wicket_pct": 0.44,
        "pace_wicket_pct": 0.56,
        "boundary_rate": 0.29,
    },
    "Himachal Pradesh Cricket Association Stadium": {
        "avg_first_innings_score": 174.0,
        "chase_win_pct": 0.44,
        "spin_wicket_pct": 0.32,
        "pace_wicket_pct": 0.68,
        "boundary_rate": 0.31,
    },
    # ---- Legacy / synthetic venue names (from dataset_generator) ----
    "Mumbai": {
        "avg_first_innings_score": 175.0,
        "chase_win_pct": 0.48,
        "spin_wicket_pct": 0.38,
        "pace_wicket_pct": 0.62,
        "boundary_rate": 0.32,
    },
    "Melbourne": {
        "avg_first_innings_score": 168.0,
        "chase_win_pct": 0.46,
        "spin_wicket_pct": 0.30,
        "pace_wicket_pct": 0.70,
        "boundary_rate": 0.28,
    },
    "Lord's": {
        "avg_first_innings_score": 155.0,
        "chase_win_pct": 0.44,
        "spin_wicket_pct": 0.28,
        "pace_wicket_pct": 0.72,
        "boundary_rate": 0.25,
    },
    "Cape Town": {
        "avg_first_innings_score": 160.0,
        "chase_win_pct": 0.43,
        "spin_wicket_pct": 0.25,
        "pace_wicket_pct": 0.75,
        "boundary_rate": 0.26,
    },
    "Auckland": {
        "avg_first_innings_score": 162.0,
        "chase_win_pct": 0.48,
        "spin_wicket_pct": 0.28,
        "pace_wicket_pct": 0.72,
        "boundary_rate": 0.27,
    },
    "Lahore": {
        "avg_first_innings_score": 170.0,
        "chase_win_pct": 0.50,
        "spin_wicket_pct": 0.45,
        "pace_wicket_pct": 0.55,
        "boundary_rate": 0.30,
    },
    "Generic Venue": {
        "avg_first_innings_score": 165.0,
        "chase_win_pct": 0.50,
        "spin_wicket_pct": 0.40,
        "pace_wicket_pct": 0.60,
        "boundary_rate": 0.29,
    },
}

DEFAULT_PROFILE: dict[str, float] = {
    "avg_first_innings_score": 165.0,
    "chase_win_pct": 0.50,
    "spin_wicket_pct": 0.40,
    "pace_wicket_pct": 0.60,
    "boundary_rate": 0.29,
    "spin_economy": 8.2,
    "pace_economy": 9.0,
}

VENUE_FEATURE_KEYS: tuple[str, ...] = (
    "avg_first_innings_score",
    "chase_win_pct",
    "spin_wicket_pct",
    "pace_wicket_pct",
    "boundary_rate",
    "spin_economy",
    "pace_economy",
)

_CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "venue_profiles.json"


def _load_live_profiles() -> dict[str, dict[str, float]]:
    """Load the cricmetric-scraped cache and merge into VENUE_PROFILES."""
    if not _CACHE_PATH.exists():
        return {}
    try:
        raw: dict[str, dict[str, float]] = json.loads(_CACHE_PATH.read_text())
        # Ensure every entry has all default keys
        return {k: {**DEFAULT_PROFILE, **v} for k, v in raw.items()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load venue cache: %s", exc)
        return {}


# Merge: static profiles are a fallback; live data takes precedence where available
_LIVE_PROFILES: dict[str, dict[str, float]] = _load_live_profiles()

# Backfill all static profiles with default spin/pace economy if missing
for _v, _p in VENUE_PROFILES.items():
    if "spin_economy" not in _p:
        _p["spin_economy"] = DEFAULT_PROFILE["spin_economy"]
    if "pace_economy" not in _p:
        _p["pace_economy"] = DEFAULT_PROFILE["pace_economy"]

_ALL_PROFILES: dict[str, dict[str, float]] = {**VENUE_PROFILES, **_LIVE_PROFILES}


def encode_venue(venue: str) -> dict[str, float]:
    """Return behavioral numeric features for *venue*.

    Checks live cricmetric cache first, then static profiles,
    then falls back to ``DEFAULT_PROFILE`` for unknown venues.
    """
    return _ALL_PROFILES.get(venue, DEFAULT_PROFILE).copy()


def get_venue_features(venue: str) -> dict[str, float]:
    """Alias for :func:`encode_venue` – preferred name in feature pipelines."""
    return encode_venue(venue)
