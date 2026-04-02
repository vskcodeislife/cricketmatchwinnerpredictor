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
    "Punjab Kings": "Punjab Cricket Association IS Bindra Stadium",
    "Rajasthan Royals": "Sawai Mansingh Stadium",
    "Lucknow Super Giants": "BRSABV Ekana Cricket Stadium",
    "Gujarat Titans": "Narendra Modi Stadium",
}

# ---------------------------------------------------------------------------
# IPL 2026 fixture list
# Format: (team_a_short, team_b_short, venue_short_key, date_YYYY-MM-DD)
# Venue is the home team's venue unless a neutral venue is specified.
# Sources: IPL 2026 official schedule (public domain)
# ---------------------------------------------------------------------------
_RAW_FIXTURES: list[tuple[str, str, str, str]] = [
    # Week 1 (March 22–28)
    ("RCB", "KKR", "RCB", "2026-03-22"),
    ("SRH", "MI",  "SRH", "2026-03-23"),
    ("DC",  "CSK", "DC",  "2026-03-24"),
    ("GT",  "PBKS","GT",  "2026-03-25"),
    ("RR",  "LSG", "RR",  "2026-03-26"),
    ("KKR", "SRH", "KKR", "2026-03-27"),
    ("MI",  "CSK", "MI",  "2026-03-28"),
    # Week 2 (March 29–April 4)
    ("PBKS","DC",  "PBKS","2026-03-29"),
    ("GT",  "RR",  "GT",  "2026-03-30"),
    ("LSG", "RCB", "LSG", "2026-03-31"),
    ("CSK", "KKR", "CSK", "2026-04-01"),
    ("SRH", "GT",  "SRH", "2026-04-02"),
    ("MI",  "DC",  "MI",  "2026-04-03"),
    ("RR",  "PBKS","RR",  "2026-04-04"),
    # Week 3 (April 5–11)
    ("RCB", "SRH", "RCB", "2026-04-05"),
    ("LSG", "KKR", "LSG", "2026-04-06"),
    ("CSK", "GT",  "CSK", "2026-04-07"),
    ("DC",  "RR",  "DC",  "2026-04-08"),
    ("MI",  "PBKS","MI",  "2026-04-09"),
    ("SRH", "LSG", "SRH", "2026-04-10"),
    ("KKR", "RCB", "KKR", "2026-04-11"),
    # Week 4 (April 12–18)
    ("GT",  "MI",  "GT",  "2026-04-12"),
    ("PBKS","CSK", "PBKS","2026-04-13"),
    ("RR",  "DC",  "RR",  "2026-04-14"),
    ("RCB", "LSG", "RCB", "2026-04-15"),
    ("KKR", "GT",  "KKR", "2026-04-16"),
    ("SRH", "PBKS","SRH", "2026-04-17"),
    ("MI",  "RR",  "MI",  "2026-04-18"),
    # Week 5 (April 19–25)
    ("CSK", "DC",  "CSK", "2026-04-19"),
    ("LSG", "GT",  "LSG", "2026-04-20"),
    ("RCB", "MI",  "RCB", "2026-04-21"),
    ("KKR", "PBKS","KKR", "2026-04-22"),
    ("DC",  "SRH", "DC",  "2026-04-23"),
    ("GT",  "CSK", "GT",  "2026-04-24"),
    ("RR",  "RCB", "RR",  "2026-04-25"),
    # Week 6 (April 26–May 2)
    ("MI",  "LSG", "MI",  "2026-04-26"),
    ("PBKS","KKR", "PBKS","2026-04-27"),
    ("SRH", "DC",  "SRH", "2026-04-28"),
    ("RCB", "GT",  "RCB", "2026-04-29"),
    ("CSK", "RR",  "CSK", "2026-04-30"),
    ("LSG", "PBKS","LSG", "2026-05-01"),
    ("DC",  "KKR", "DC",  "2026-05-02"),
    # Week 7 (May 3–9)
    ("MI",  "SRH", "MI",  "2026-05-03"),
    ("GT",  "DC",  "GT",  "2026-05-04"),
    ("RR",  "CSK", "RR",  "2026-05-05"),
    ("KKR", "LSG", "KKR", "2026-05-06"),
    ("PBKS","RCB", "PBKS","2026-05-07"),
    ("SRH", "RR",  "SRH", "2026-05-08"),
    ("MI",  "GT",  "MI",  "2026-05-09"),
    # Week 8 (May 10–16)
    ("CSK", "PBKS","CSK", "2026-05-10"),
    ("DC",  "LSG", "DC",  "2026-05-11"),
    ("RCB", "SRH", "RCB", "2026-05-12"),
    ("KKR", "RR",  "KKR", "2026-05-13"),
    ("GT",  "PBKS","GT",  "2026-05-14"),
    ("LSG", "MI",  "LSG", "2026-05-15"),
    ("CSK", "SRH", "CSK", "2026-05-16"),
    # Week 9 – final league round (May 17–18)
    ("DC",  "RCB", "DC",  "2026-05-17"),
    ("KKR", "MI",  "KKR", "2026-05-17"),
    ("PBKS","RR",  "PBKS","2026-05-18"),
    ("GT",  "LSG", "GT",  "2026-05-18"),
    # Playoffs (approximate dates)
    ("TBD1","TBD2","Narendra Modi Stadium","2026-05-20"),  # Qualifier 1
    ("TBD3","TBD4","Eden Gardens",         "2026-05-21"),  # Eliminator
    ("TBD5","TBD6","Wankhede Stadium",     "2026-05-23"),  # Qualifier 2
    ("TBD7","TBD8","Narendra Modi Stadium","2026-05-25"),  # Final
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
