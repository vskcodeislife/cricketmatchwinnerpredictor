"""Venue behavioral profiles.

At runtime, the computed venue cache (``data/venue_profiles.json``) provides
real profiles derived from cricsheet ball-by-ball data.  A small static
fallback table covers synthetic/legacy venue names so offline training works.

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
    Mean economy rate of spin bowlers.
pace_economy
    Mean economy rate of pace bowlers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static fallback for synthetic / legacy venue names only
# ---------------------------------------------------------------------------
VENUE_PROFILES: dict[str, dict[str, float]] = {
    "Mumbai": {
        "avg_first_innings_score": 174.0,
        "chase_win_pct": 0.55,
        "spin_wicket_pct": 0.20,
        "pace_wicket_pct": 0.80,
        "boundary_rate": 0.18,
    },
    "Melbourne": {
        "avg_first_innings_score": 168.0,
        "chase_win_pct": 0.46,
        "spin_wicket_pct": 0.18,
        "pace_wicket_pct": 0.82,
        "boundary_rate": 0.16,
    },
    "Lord's": {
        "avg_first_innings_score": 155.0,
        "chase_win_pct": 0.44,
        "spin_wicket_pct": 0.15,
        "pace_wicket_pct": 0.85,
        "boundary_rate": 0.14,
    },
    "Cape Town": {
        "avg_first_innings_score": 160.0,
        "chase_win_pct": 0.43,
        "spin_wicket_pct": 0.14,
        "pace_wicket_pct": 0.86,
        "boundary_rate": 0.15,
    },
    "Auckland": {
        "avg_first_innings_score": 162.0,
        "chase_win_pct": 0.48,
        "spin_wicket_pct": 0.16,
        "pace_wicket_pct": 0.84,
        "boundary_rate": 0.15,
    },
    "Lahore": {
        "avg_first_innings_score": 170.0,
        "chase_win_pct": 0.50,
        "spin_wicket_pct": 0.30,
        "pace_wicket_pct": 0.70,
        "boundary_rate": 0.17,
    },
    "Generic Venue": {
        "avg_first_innings_score": 165.0,
        "chase_win_pct": 0.50,
        "spin_wicket_pct": 0.22,
        "pace_wicket_pct": 0.78,
        "boundary_rate": 0.17,
    },
}

DEFAULT_PROFILE: dict[str, float] = {
    "avg_first_innings_score": 168.0,
    "chase_win_pct": 0.50,
    "spin_wicket_pct": 0.20,
    "pace_wicket_pct": 0.80,
    "boundary_rate": 0.17,
    "spin_economy": 7.6,
    "pace_economy": 8.4,
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


def _load_computed_profiles() -> dict[str, dict[str, float]]:
    """Load precomputed venue profiles from cricsheet ball-by-ball data."""
    if not _CACHE_PATH.exists():
        return {}
    try:
        raw: dict[str, dict[str, float]] = json.loads(_CACHE_PATH.read_text())
        # Ensure every entry has all default keys
        return {k: {**DEFAULT_PROFILE, **v} for k, v in raw.items()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load venue cache: %s", exc)
        return {}


# Merge: computed profiles take precedence; static profiles are fallback only
_COMPUTED_PROFILES: dict[str, dict[str, float]] = _load_computed_profiles()

# Backfill all static profiles with default spin/pace economy if missing
for _v, _p in VENUE_PROFILES.items():
    if "spin_economy" not in _p:
        _p["spin_economy"] = DEFAULT_PROFILE["spin_economy"]
    if "pace_economy" not in _p:
        _p["pace_economy"] = DEFAULT_PROFILE["pace_economy"]

_ALL_PROFILES: dict[str, dict[str, float]] = {**VENUE_PROFILES, **_COMPUTED_PROFILES}


def encode_venue(venue: str) -> dict[str, float]:
    """Return behavioral numeric features for *venue*.

    Checks precomputed profiles first (from cricsheet data), then static
    fallback profiles, then falls back to ``DEFAULT_PROFILE`` for unknown venues.
    """
    return _ALL_PROFILES.get(venue, DEFAULT_PROFILE).copy()


def get_venue_features(venue: str) -> dict[str, float]:
    """Alias for :func:`encode_venue` – preferred name in feature pipelines."""
    return encode_venue(venue)
