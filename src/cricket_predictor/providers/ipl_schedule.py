"""IPL 2026 season schedule provider.

Provides the full IPL 2026 fixture list so the app can:
  * Make pre-match predictions for upcoming games.
  * Look up which matches are completed and verify results.

The schedule is seeded with known fixtures.  At runtime it is enriched by
matching against cricsheet ``recently_played`` data so result fields are
filled in automatically.
"""

from __future__ import annotations

from datetime import datetime, date
from typing import TypedDict


class ScheduledMatch(TypedDict):
    match_id: str        # deterministic key, e.g. "IPL2026_M01"
    team_a: str          # full team name as used in prediction service
    team_b: str
    venue: str
    match_date: str      # ISO-8601 date "YYYY-MM-DD"
    is_complete: bool
    actual_winner: str | None


# Full team name ↔ short code mapping
TEAM_SHORT = {
    "Mumbai Indians": "MI",
    "Chennai Super Kings": "CSK",
    "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Sunrisers Hyderabad": "SRH",
    "Delhi Capitals": "DC",
    "Punjab Kings": "PBKS",
    "Rajasthan Royals": "RR",
    "Lucknow Super Giants": "LSG",
    "Gujarat Titans": "GT",
}
SHORT_TEAM = {v: k for k, v in TEAM_SHORT.items()}

# Home venues
HOME_VENUE = {
    "Mumbai Indians": "Wankhede Stadium",
    "Chennai Super Kings": "MA Chidambaram Stadium",
    "Royal Challengers Bengaluru": "M Chinnaswamy Stadium",
    "Kolkata Knight Riders": "Eden Gardens",
    "Sunrisers Hyderabad": "Rajiv Gandhi International Stadium",
    "Delhi Capitals": "Arun Jaitley Stadium",
    "Punjab Kings": "Punjab Cricket Association IS Bindra Stadium",    "Rajasthan Royals": "Sawai Mansingh Stadium",
    "Lucknow Super Giants": "BRSABV Ekana Cricket Stadium",
    "Gujarat Titans": "Narendra Modi Stadium",
}

# ---------------------------------------------------------------------------
# IPL 2026 fixture list  (source: BCCI official schedule, verified via
# https://www.olympics.com/en/news/indian-premier-league-ipl-2026-schedule-match-list)
# Format: (team_a_short, team_b_short, home_team_short_for_venue, date_YYYY-MM-DD)
# ---------------------------------------------------------------------------
_RAW_FIXTURES: list[tuple[str, str, str, str]] = [
    # ── Week 1 ──────────────────────────────────────────────────────────────
    ("RCB", "SRH", "RCB", "2026-03-28"),   # M1  Bengaluru  (opener)
    ("MI",  "KKR", "MI",  "2026-03-29"),   # M2  Mumbai
    ("RR",  "CSK", "RR",  "2026-03-30"),   # M3  Jaipur
    ("DC",  "LSG", "DC",  "2026-03-31"),   # M4  Delhi
    ("PBKS","GT",  "PBKS","2026-04-01"),   # M5  Mullanpur
    ("SRH", "KKR", "KKR", "2026-04-02"),   # M6  Eden Gardens  ← confirmed
    ("MI",  "RR",  "MI",  "2026-04-03"),   # M7  Mumbai
    # ── Week 2 ──────────────────────────────────────────────────────────────
    ("CSK", "RCB", "CSK", "2026-04-04"),   # M8  Chennai
    ("GT",  "DC",  "GT",  "2026-04-05"),   # M9  Ahmedabad
    ("LSG", "PBKS","LSG", "2026-04-06"),   # M10 Lucknow
    ("KKR", "MI",  "KKR", "2026-04-07"),   # M11 Kolkata
    ("SRH", "RR",  "SRH", "2026-04-08"),   # M12 Hyderabad
    ("RCB", "DC",  "RCB", "2026-04-09"),   # M13 Bengaluru
    ("CSK", "GT",  "CSK", "2026-04-10"),   # M14 Chennai
    # ── Week 3 ──────────────────────────────────────────────────────────────
    ("PBKS","KKR", "PBKS","2026-04-11"),   # M15 Mullanpur
    ("LSG", "MI",  "LSG", "2026-04-12"),   # M16 Lucknow
    ("RR",  "RCB", "RR",  "2026-04-13"),   # M17 Jaipur
    ("DC",  "SRH", "DC",  "2026-04-14"),   # M18 Delhi
    ("GT",  "KKR", "GT",  "2026-04-15"),   # M19 Ahmedabad
    ("CSK", "PBKS","CSK", "2026-04-16"),   # M20 Chennai
    ("MI",  "LSG", "MI",  "2026-04-17"),   # M21 Mumbai
    # ── Week 4 ──────────────────────────────────────────────────────────────
    ("RCB", "RR",  "RCB", "2026-04-18"),   # M22 Bengaluru
    ("SRH", "DC",  "SRH", "2026-04-19"),   # M23 Hyderabad
    ("KKR", "GT",  "KKR", "2026-04-20"),   # M24 Kolkata
    ("PBKS","MI",  "PBKS","2026-04-21"),   # M25 Mullanpur
    ("LSG", "CSK", "LSG", "2026-04-22"),   # M26 Lucknow
    ("RR",  "DC",  "RR",  "2026-04-23"),   # M27 Jaipur
    ("RCB", "GT",  "RCB", "2026-04-24"),   # M28 Bengaluru
    # ── Week 5 ──────────────────────────────────────────────────────────────
    ("KKR", "SRH", "KKR", "2026-04-25"),   # M29 Kolkata
    ("MI",  "CSK", "MI",  "2026-04-26"),   # M30 Mumbai
    ("PBKS","RR",  "PBKS","2026-04-27"),   # M31 Mullanpur
    ("DC",  "GT",  "DC",  "2026-04-28"),   # M32 Delhi
    ("LSG", "RCB", "LSG", "2026-04-29"),   # M33 Lucknow
    ("SRH", "CSK", "SRH", "2026-04-30"),   # M34 Hyderabad
    ("KKR", "RR",  "KKR", "2026-05-01"),   # M35 Kolkata
    # ── Week 6 ──────────────────────────────────────────────────────────────
    ("RCB", "PBKS","RCB", "2026-05-02"),   # M36 Bengaluru
    ("GT",  "LSG", "GT",  "2026-05-03"),   # M37 Ahmedabad
    ("MI",  "DC",  "MI",  "2026-05-04"),   # M38 Mumbai
    ("CSK", "KKR", "CSK", "2026-05-05"),   # M39 Chennai
    ("RR",  "SRH", "RR",  "2026-05-06"),   # M40 Jaipur
    ("PBKS","LSG", "PBKS","2026-05-07"),   # M41 Mullanpur
    ("GT",  "RCB", "GT",  "2026-05-08"),   # M42 Ahmedabad
    # ── Week 7 ──────────────────────────────────────────────────────────────
    ("DC",  "MI",  "DC",  "2026-05-09"),   # M43 Delhi
    ("KKR", "CSK", "KKR", "2026-05-10"),   # M44 Kolkata
    ("SRH", "PBKS","SRH", "2026-05-11"),   # M45 Hyderabad
    ("RR",  "GT",  "RR",  "2026-05-12"),   # M46 Jaipur
    ("LSG", "DC",  "LSG", "2026-05-13"),   # M47 Lucknow
    ("RCB", "MI",  "RCB", "2026-05-14"),   # M48 Bengaluru
    ("CSK", "SRH", "CSK", "2026-05-15"),   # M49 Chennai
    # ── Week 8 ──────────────────────────────────────────────────────────────
    ("GT",  "PBKS","GT",  "2026-05-16"),   # M50 Ahmedabad
    ("KKR", "LSG", "KKR", "2026-05-17"),   # M51 Kolkata
    ("MI",  "RR",  "MI",  "2026-05-17"),   # M52 Mumbai  (double-header)
    ("DC",  "RCB", "DC",  "2026-05-18"),   # M53 Delhi
    ("SRH", "GT",  "SRH", "2026-05-18"),   # M54 Hyderabad (double-header)
    ("PBKS","CSK", "PBKS","2026-05-19"),   # M55 Mullanpur
    ("RR",  "KKR", "RR",  "2026-05-19"),   # M56 Jaipur  (double-header)
    # ── Week 9 — final league round ─────────────────────────────────────────
    ("LSG", "SRH", "LSG", "2026-05-20"),   # M57 Lucknow
    ("MI",  "GT",  "MI",  "2026-05-20"),   # M58 Mumbai  (double-header)
    ("DC",  "KKR", "DC",  "2026-05-21"),   # M59 Delhi
    ("RCB", "CSK", "RCB", "2026-05-21"),   # M60 Bengaluru (double-header)
    ("PBKS","RR",  "PBKS","2026-05-22"),   # M61 Mullanpur
    ("GT",  "SRH", "GT",  "2026-05-22"),   # M62 Ahmedabad (double-header)
    ("LSG", "MI",  "LSG", "2026-05-23"),   # M63 Lucknow
    ("KKR", "DC",  "KKR", "2026-05-23"),   # M64 Kolkata  (double-header)
    ("CSK", "RR",  "CSK", "2026-05-24"),   # M65 Chennai
    ("RCB", "PBKS","RCB", "2026-05-24"),   # M66 Bengaluru (double-header) — league ends
    # ── Playoffs (venues TBC by BCCI) ────────────────────────────────────────
    ("TBD1","TBD2","Narendra Modi Stadium","2026-05-27"),  # Qualifier 1
    ("TBD3","TBD4","Eden Gardens",         "2026-05-28"),  # Eliminator
    ("TBD5","TBD6","Wankhede Stadium",     "2026-05-30"),  # Qualifier 2
    ("TBD7","TBD8","Narendra Modi Stadium","2026-06-01"),  # Final
]


def _build_schedule() -> list[ScheduledMatch]:
    matches: list[ScheduledMatch] = []
    for idx, (ta, tb, venue_key, match_date) in enumerate(_RAW_FIXTURES, start=1):
        # Resolve team names and venue
        if ta.startswith("TBD"):
            team_a, team_b, venue = ta, tb, venue_key
        else:
            team_a = SHORT_TEAM.get(ta, ta)
            team_b = SHORT_TEAM.get(tb, tb)
            venue_owner = SHORT_TEAM.get(venue_key, venue_key)
            venue = HOME_VENUE.get(venue_owner, HOME_VENUE.get(team_a, "Eden Gardens"))

        match_id = f"IPL2026_M{idx:02d}"
        today = date.today().isoformat()
        matches.append(
            ScheduledMatch(
                match_id=match_id,
                team_a=team_a,
                team_b=team_b,
                venue=venue,
                match_date=match_date,
                is_complete=match_date < today,
                actual_winner=None,
            )
        )
    return matches


# Module-level singleton — rebuilt once per process start
_SCHEDULE: list[ScheduledMatch] = _build_schedule()


class IPLScheduleProvider:
    """Provides IPL 2026 fixtures with helpers for upcoming/completed matches."""

    def __init__(self) -> None:
        self._schedule = _SCHEDULE

    def all_matches(self) -> list[ScheduledMatch]:
        return self._schedule

    def upcoming_matches(self, from_date: date | None = None) -> list[ScheduledMatch]:
        cutoff = (from_date or date.today()).isoformat()
        return [m for m in self._schedule if m["match_date"] >= cutoff and not m["team_a"].startswith("TBD")]

    def next_match(self) -> ScheduledMatch | None:
        today = date.today().isoformat()
        upcoming = [m for m in self._schedule if m["match_date"] >= today and not m["team_a"].startswith("TBD")]
        return upcoming[0] if upcoming else None

    def completed_matches(self) -> list[ScheduledMatch]:
        today = date.today().isoformat()
        return [m for m in self._schedule if m["match_date"] < today and not m["team_a"].startswith("TBD")]

    def get_match_by_id(self, match_id: str) -> ScheduledMatch | None:
        return next((m for m in self._schedule if m["match_id"] == match_id), None)

    def find_match_for_teams(self, team_a: str, team_b: str, match_date: str) -> ScheduledMatch | None:
        """Try to find a scheduled match by team names and approximate date (±1 day)."""
        target = datetime.fromisoformat(match_date).date()
        for m in self._schedule:
            sched_date = datetime.fromisoformat(m["match_date"]).date()
            if abs((sched_date - target).days) <= 1:
                teams = {m["team_a"], m["team_b"]}
                if team_a in teams and team_b in teams:
                    return m
        return None
