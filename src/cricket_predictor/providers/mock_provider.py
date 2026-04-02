from __future__ import annotations

from typing import Any

from cricket_predictor.providers.base import LiveDataProvider


class MockLiveDataProvider(LiveDataProvider):
    async def fetch_live_match_context(self) -> list[dict[str, Any]]:
        return [
            {
                "team_a": "India",
                "team_b": "Australia",
                "venue": "Mumbai",
                "match_format": "ODI",
                "pitch_type": "batting",
                "toss_winner": "India",
                "toss_decision": "bat",
                "team_a_recent_form": 0.84,
                "team_b_recent_form": 0.74,
                "team_a_batting_strength": 89.0,
                "team_b_batting_strength": 84.0,
                "team_a_bowling_strength": 80.0,
                "team_b_bowling_strength": 78.0,
                "head_to_head_win_pct_team_a": 0.58,
                "venue_advantage_team_a": 1.0,
            }
        ]
