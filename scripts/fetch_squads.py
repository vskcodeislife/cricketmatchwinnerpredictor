"""Fetch and cache IPL 2026 squad data from iplt20.com.

Usage
-----
  python scripts/fetch_squads.py

Writes to data/squad_profiles.json.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    from cricket_predictor.providers.ipl_squad_provider import IPLSquadProvider

    provider = IPLSquadProvider()
    log.info("Fetching IPL 2026 squads from iplt20.com …")
    squads = provider.fetch_all()
    path = provider.save(squads)

    print(f"\nSquad cache saved → {path}\n")
    print(f"{'Team':<32} {'Total':>5}  {'Bat':>3}  {'AR':>3}  {'Bowl':>4}")
    print("-" * 55)
    for team, data in squads.items():
        total = data.get("total", 0)
        bat   = len(data.get("batters", []))
        ar    = len(data.get("all_rounders", []))
        bowl  = len(data.get("bowlers", []))
        err   = "  ⚠ " + data["error"] if "error" in data else ""
        print(f"{team:<32} {total:>5}  {bat:>3}  {ar:>3}  {bowl:>4}{err}")


if __name__ == "__main__":
    main()
