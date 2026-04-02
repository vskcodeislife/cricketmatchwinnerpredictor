"""Injury / availability scraper for crictracker.com.

Fetches https://www.crictracker.com/cricket-appeal/ipl-2026-full-list-of-injured-and-unavailable-players/
and extracts the table of injured/unavailable players.  Each row has:

    Sr. No. | Player | Franchise | Role | Injury/Reason

Results are cached to ``data/injury_report.json`` and auto-fed into the
match override system so the prediction tracker picks them up.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_URL = "https://www.crictracker.com/cricket-appeal/ipl-2026-full-list-of-injured-and-unavailable-players/"
_DEFAULT_CACHE = Path(__file__).parents[3] / "data" / "injury_report.json"


# ---------------------------------------------------------------------------
# HTML parsing — extract table rows
# ---------------------------------------------------------------------------

def _parse_injury_table(html: str) -> list[dict[str, str]]:
    """Extract injury table rows from the article HTML.

    The table is standard HTML: <table><thead>...<tbody><tr><td>...</td>...
    Columns: Sr. No. | Player | Franchise | Role | Injury/Reason
    """
    entries: list[dict[str, str]] = []

    # Find all table rows (skip header)
    row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
    cell_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)

    rows = row_pattern.findall(html)
    for row in rows:
        cells = cell_pattern.findall(row)
        if len(cells) < 5:
            continue
        # Strip HTML tags from cell content
        clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        # Skip header row
        if clean[0].lower().startswith("sr"):
            continue
        # Must have a numeric first column
        if not clean[0].isdigit():
            continue

        entries.append({
            "sr_no": int(clean[0]),
            "player": clean[1],
            "franchise": clean[2],
            "role": clean[3],
            "reason": clean[4],
        })

    return entries


def _classify_availability(reason: str) -> str:
    """Classify injury reason into availability status."""
    lower = reason.lower()
    if "out of season" in lower or "ruled out" in lower:
        return "out_for_season"
    if "doubtful" in lower:
        return "doubtful"
    if "early games" in lower or "first two weeks" in lower:
        return "miss_early_games"
    if "opted out" in lower:
        return "opted_out"
    return "unavailable"


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class InjuryReportProvider:
    """Fetches and caches the IPL 2026 injury/availability report."""

    def __init__(
        self,
        url: str = _URL,
        cache_path: str | Path = _DEFAULT_CACHE,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._url = url
        self._cache_path = Path(cache_path)
        self._timeout = timeout_seconds

    def fetch(self) -> dict[str, Any]:
        """Fetch the injury page, parse, and return structured data."""
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
            "Accept": "text/html,*/*;q=0.8",
        }

        resp = httpx.get(self._url, headers=headers, timeout=self._timeout, follow_redirects=True)
        resp.raise_for_status()

        entries = _parse_injury_table(resp.text)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Group by franchise
        by_team: dict[str, list] = {}
        for e in entries:
            e["status"] = _classify_availability(e["reason"])
            by_team.setdefault(e["franchise"], []).append(e)

        return {
            "fetched_at": now,
            "source_url": self._url,
            "total_unavailable": len(entries),
            "players": entries,
            "by_team": by_team,
        }

    def save(self, report: dict[str, Any] | None = None) -> Path:
        """Fetch (if not provided) and write cache file."""
        if report is None:
            report = self.fetch()
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(report, indent=2))
        log.info("Injury report saved to %s (%d players)", self._cache_path, report["total_unavailable"])
        return self._cache_path

    def load(self) -> dict[str, Any]:
        """Return cached report, or empty dict if cache does not exist."""
        if not self._cache_path.exists():
            return {}
        return json.loads(self._cache_path.read_text())

    def build_override_text(self, report: dict[str, Any] | None = None) -> str:
        """Convert the injury report into override text the parser can consume.

        Output example:
            Pat Cummins is injured. Harshit Rana is out.
        """
        if report is None:
            report = self.load()
        lines: list[str] = []
        for entry in report.get("players", []):
            status = entry.get("status", "unavailable")
            player = entry["player"]
            if status == "out_for_season":
                lines.append(f"{player} is out for the season")
            elif status == "doubtful":
                lines.append(f"{player} is doubtful")
            elif status == "miss_early_games":
                lines.append(f"{player} is unavailable for early games")
            elif status == "opted_out":
                lines.append(f"{player} is unavailable")
            else:
                lines.append(f"{player} is injured")
        return ". ".join(lines)
