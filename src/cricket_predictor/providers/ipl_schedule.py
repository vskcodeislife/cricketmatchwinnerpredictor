"""IPL 2026 season schedule provider.

Provides the full IPL 2026 fixture list so the app can:
  * Make pre-match predictions for upcoming games.
  * Look up which matches are completed and verify results.

The schedule is seeded with known fixtures.  At runtime it is enriched by
matching against cricsheet ``recently_played`` data so result fields are
filled in automatically.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta, timezone
from typing import TypedDict

# IST = UTC+5:30
_IST = timezone(timedelta(hours=5, minutes=30))

# Match start hour mapping
_TIME_HOURS = {
    "3:00 PM IST": 15,
    "7:30 PM IST": 19,
}


class ScheduledMatch(TypedDict):
    match_id: str        # deterministic key, e.g. "IPL2026_M01"
    team_a: str          # full team name as used in prediction service
    team_b: str
    venue: str
    match_date: str      # ISO-8601 date "YYYY-MM-DD"
    match_time: str      # e.g. "3:00 PM IST" or "7:30 PM IST"
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

# Neutral / secondary venue lookup (city → venue name)
_CITY_VENUE = {
    "Hyderabad":       "Rajiv Gandhi International Stadium",
    "Chennai":         "MA Chidambaram Stadium",
    "Delhi":           "Arun Jaitley Stadium",
    "Chandigarh":      "Punjab Cricket Association IS Bindra Stadium",
    "New Chandigarh":  "Punjab Cricket Association IS Bindra Stadium",
    "Lucknow":         "BRSABV Ekana Cricket Stadium",
    "Kolkata":         "Eden Gardens",
    "Bengaluru":       "M Chinnaswamy Stadium",
    "Guwahati":        "ACA Stadium, Barsapara",
    "Mumbai":          "Wankhede Stadium",
    "Ahmedabad":       "Narendra Modi Stadium",
    "Jaipur":          "Sawai Mansingh Stadium",
    "Dharamshala":     "HPCA Stadium, Dharamshala",
}

# ---------------------------------------------------------------------------
# IPL 2026 league fixture list from the user-provided schedule.
# Format: (home_team_short, away_team_short, venue_city, date_YYYY-MM-DD)
# ---------------------------------------------------------------------------
_RAW_FIXTURES: list[tuple[str, str, str, str]] = [
    ("RCB", "SRH", "Bengaluru", "2026-03-28"),
    ("MI", "KKR", "Mumbai", "2026-03-29"),
    ("RR", "CSK", "Guwahati", "2026-03-30"),
    ("PBKS", "GT", "Chandigarh", "2026-03-31"),
    ("LSG", "DC", "Lucknow", "2026-04-01"),
    ("KKR", "SRH", "Kolkata", "2026-04-02"),
    ("CSK", "PBKS", "Chennai", "2026-04-03"),
    ("DC", "MI", "Delhi", "2026-04-04"),
    ("GT", "RR", "Ahmedabad", "2026-04-04"),
    ("SRH", "LSG", "Hyderabad", "2026-04-05"),
    ("RCB", "CSK", "Bengaluru", "2026-04-05"),
    ("KKR", "PBKS", "Kolkata", "2026-04-06"),
    ("RR", "MI", "Guwahati", "2026-04-07"),
    ("DC", "GT", "Delhi", "2026-04-08"),
    ("KKR", "LSG", "Kolkata", "2026-04-09"),
    ("RR", "RCB", "Guwahati", "2026-04-10"),
    ("PBKS", "SRH", "Chandigarh", "2026-04-11"),
    ("CSK", "DC", "Chennai", "2026-04-11"),
    ("LSG", "GT", "Lucknow", "2026-04-12"),
    ("MI", "RCB", "Mumbai", "2026-04-12"),
    ("SRH", "CSK", "Hyderabad", "2026-04-13"),
    ("DC", "KKR", "Delhi", "2026-04-14"),
    ("RCB", "LSG", "Bengaluru", "2026-04-15"),
    ("GT", "MI", "Ahmedabad", "2026-04-16"),
    ("PBKS", "RR", "Chandigarh", "2026-04-17"),
    ("RCB", "DC", "Bengaluru", "2026-04-18"),
    ("CSK", "LSG", "Chennai", "2026-04-18"),
    ("MI", "SRH", "Mumbai", "2026-04-19"),
    ("GT", "KKR", "Ahmedabad", "2026-04-19"),
    ("RR", "DC", "Jaipur", "2026-04-20"),
    ("LSG", "PBKS", "Lucknow", "2026-04-21"),
    ("SRH", "GT", "Hyderabad", "2026-04-22"),
    ("CSK", "MI", "Chennai", "2026-04-23"),
    ("RCB", "GT", "Bengaluru", "2026-04-24"),
    ("KKR", "RR", "Kolkata", "2026-04-25"),
    ("PBKS", "DC", "Chandigarh", "2026-04-25"),
    ("MI", "LSG", "Mumbai", "2026-04-26"),
    ("SRH", "RCB", "Hyderabad", "2026-04-26"),
    ("GT", "CSK", "Ahmedabad", "2026-04-27"),
    ("RR", "KKR", "Jaipur", "2026-04-28"),
    ("MI", "SRH", "Mumbai", "2026-04-29"),
    ("GT", "RCB", "Ahmedabad", "2026-04-30"),
    ("DC", "CSK", "Delhi", "2026-05-01"),
    ("PBKS", "MI", "Chandigarh", "2026-05-02"),
    ("LSG", "RR", "Lucknow", "2026-05-03"),
    ("KKR", "RCB", "Kolkata", "2026-05-03"),
    ("SRH", "DC", "Hyderabad", "2026-05-04"),
    ("CSK", "GT", "Chennai", "2026-05-05"),
    ("MI", "RR", "Mumbai", "2026-05-06"),
    ("PBKS", "LSG", "Chandigarh", "2026-05-07"),
    ("RCB", "SRH", "Bengaluru", "2026-05-08"),
    ("DC", "KKR", "Delhi", "2026-05-09"),
    ("RR", "GT", "Jaipur", "2026-05-10"),
    ("LSG", "CSK", "Lucknow", "2026-05-10"),
    ("MI", "PBKS", "Mumbai", "2026-05-11"),
    ("SRH", "KKR", "Hyderabad", "2026-05-12"),
    ("GT", "DC", "Ahmedabad", "2026-05-13"),
    ("CSK", "RR", "Chennai", "2026-05-14"),
    ("RCB", "PBKS", "Bengaluru", "2026-05-15"),
    ("KKR", "MI", "Kolkata", "2026-05-16"),
    ("LSG", "SRH", "Lucknow", "2026-05-16"),
    ("DC", "RR", "Delhi", "2026-05-17"),
    ("GT", "PBKS", "Ahmedabad", "2026-05-17"),
    ("CSK", "KKR", "Chennai", "2026-05-18"),
    ("RCB", "MI", "Bengaluru", "2026-05-19"),
    ("SRH", "RR", "Hyderabad", "2026-05-20"),
    ("PBKS", "CSK", "Chandigarh", "2026-05-21"),
    ("MI", "GT", "Mumbai", "2026-05-22"),
    ("KKR", "DC", "Kolkata", "2026-05-23"),
    ("RR", "LSG", "Jaipur", "2026-05-24"),
]


def _build_schedule() -> list[ScheduledMatch]:
    # Pre-scan to find dates with double-headers
    from collections import Counter
    date_counts = Counter(d for _, _, _, d in _RAW_FIXTURES)
    date_seen: dict[str, int] = {}  # track which match on a given date

    matches: list[ScheduledMatch] = []
    for idx, (ta, tb, venue_city, match_date) in enumerate(_RAW_FIXTURES, start=1):
        # Resolve team names and venue
        if ta.startswith("TBD"):
            team_a, team_b, venue = ta, tb, venue_city
        else:
            team_a = SHORT_TEAM.get(ta, ta)
            team_b = SHORT_TEAM.get(tb, tb)
            venue = _CITY_VENUE.get(venue_city, HOME_VENUE.get(team_a, venue_city))

        # Assign match time: double-header → 1st at 3:00 PM, 2nd at 7:30 PM
        occurrence = date_seen.get(match_date, 0) + 1
        date_seen[match_date] = occurrence
        if date_counts[match_date] >= 2 and occurrence == 1:
            match_time = "3:00 PM IST"
        else:
            match_time = "7:30 PM IST"

        match_id = f"IPL2026_M{idx:02d}"
        today = date.today().isoformat()
        matches.append(
            ScheduledMatch(
                match_id=match_id,
                team_a=team_a,
                team_b=team_b,
                venue=venue,
                match_date=match_date,
                match_time=match_time,
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
        now_ist = datetime.now(_IST)
        today_str = now_ist.date().isoformat()
        current_hour = now_ist.hour
        upcoming: list[ScheduledMatch] = []
        for m in self._schedule:
            if m["team_a"].startswith("TBD"):
                continue
            if m["match_date"] > today_str:
                upcoming.append(m)
            elif m["match_date"] == today_str:
                # Include only if the match hasn't started yet
                start_hour = _TIME_HOURS.get(m["match_time"], 19)
                if current_hour < start_hour:
                    upcoming.append(m)
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
