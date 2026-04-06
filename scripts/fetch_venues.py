"""Compute venue profiles from cricsheet ball-by-ball data and save to data/venue_profiles.json."""
from __future__ import annotations

import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from cricket_predictor.data.cricsheet_loader import compute_venue_profiles

_CACHE_PATH = Path("data/venue_profiles.json")

# Only generate profiles for venues that host IPL or major T20 matches.
# Global associate T20 venues are excluded to keep the model focused.
_IPL_VENUE_NAMES: set[str] = {
    "Arun Jaitley Stadium",
    "Eden Gardens",
    "Wankhede Stadium",
    "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium",
    "Rajiv Gandhi International Stadium",
    "Narendra Modi Stadium",
    "Sawai Mansingh Stadium",
    "Punjab Cricket Association IS Bindra Stadium",
    "BRSABV Ekana Cricket Stadium",
    "Himachal Pradesh Cricket Association Stadium",
    "DY Patil Stadium",
    "Brabourne Stadium",
    "Maharashtra Cricket Association Stadium",
    "ACA-VDCA Cricket Stadium",
    "Holkar Cricket Stadium",
    "Saurashtra Cricket Association Stadium",
    "Barsapara Cricket Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium",
    "Raipur International Cricket Stadium",
    "JSCA International Stadium Complex",
    "Barabati Stadium",
    "Green Park",
    # International T20 venues that appear in cricsheet data
    "Dubai International Cricket Stadium",
    "Sheikh Zayed Stadium",
    "Sharjah Cricket Stadium",
    "Kennington Oval",
    "Edgbaston",
    "Adelaide Oval",
    "Melbourne Cricket Ground",
    "Sydney Cricket Ground",
    "Eden Park",
    "Hagley Oval",
    "Gaddafi Stadium",
    "Harare Sports Club",
    "Kensington Oval",
}


if __name__ == "__main__":
    data_dir = Path("data/cricsheet")
    json_dirs = [p for p in data_dir.iterdir() if p.is_dir()] if data_dir.exists() else []
    if not json_dirs:
        print(f"No JSON directories found under {data_dir}")
        sys.exit(1)

    print(f"Computing venue profiles from {len(json_dirs)} directories …")
    all_profiles = compute_venue_profiles(json_dirs, min_matches=3)

    # Filter to IPL / major venues only
    profiles = {k: v for k, v in all_profiles.items() if k in _IPL_VENUE_NAMES}

    # Remove the internal 'matches' count before saving
    for venue, profile in profiles.items():
        matches = profile.pop("matches", 0)
        print(f"  {venue}: {matches} matches, avg 1st inn = {profile['avg_first_innings_score']}")

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(profiles, indent=2) + "\n")
    print(f"\nSaved {len(profiles)} venue profiles to {_CACHE_PATH}")
    print(f"(Skipped {len(all_profiles) - len(profiles)} non-IPL venues)")
