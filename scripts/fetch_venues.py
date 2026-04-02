"""One-time script to fetch all IPL venue profiles from cricmetric and cache them."""
from __future__ import annotations

import sys
import logging

sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

try:
    import truststore  # noqa: F401
    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

from cricket_predictor.providers.cricmetric_venue import CricmetricVenueProvider

ALREADY_FETCHED = {
    "Eden Gardens": {
        "avg_first_innings_score": 181.71,
        "chase_win_pct": 0.5,
        "spin_wicket_pct": 0.8166,
        "pace_wicket_pct": 0.1834,
        "boundary_rate": 0.4,
        "spin_economy": 6.558,
        "pace_economy": 8.382,
    },
    "Wankhede Stadium": {
        "avg_first_innings_score": 184.25,
        "chase_win_pct": 0.4667,
        "spin_wicket_pct": 0.7104,
        "pace_wicket_pct": 0.2896,
        "boundary_rate": 0.4,
        "spin_economy": 8.5692,
        "pace_economy": 9.6423,
    },
    "MA Chidambaram Stadium": {
        "avg_first_innings_score": 197.0,
        "chase_win_pct": 0.5556,
        "spin_wicket_pct": 0.6669,
        "pace_wicket_pct": 0.3331,
        "boundary_rate": 0.4,
        "spin_economy": 8.1195,
        "pace_economy": 10.1393,
    },
    "M Chinnaswamy Stadium": {
        "avg_first_innings_score": 212.0,
        "chase_win_pct": 0.6667,
        "spin_wicket_pct": 0.7378,
        "pace_wicket_pct": 0.2622,
        "boundary_rate": 0.4,
        "spin_economy": 7.553,
        "pace_economy": 8.8617,
    },
}

if __name__ == "__main__":
    provider = CricmetricVenueProvider()
    provider._cache.update(ALREADY_FETCHED)
    provider.fetch_all_ipl_venues(delay_secs=1.2)
    for name, prof in sorted(provider._cache.items()):
        print(f"  {name}: {prof}")
