from __future__ import annotations

import pandas as pd

from cricket_predictor.features.venue_encoder import get_venue_features


MATCH_FEATURE_COLUMNS = [
    "venue",
    "match_format",
    "pitch_type",
    "toss_winner",
    "toss_decision",
    "team_a_recent_form",
    "team_b_recent_form",
    "team_a_batting_strength",
    "team_b_batting_strength",
    "team_a_bowling_strength",
    "team_b_bowling_strength",
    "head_to_head_win_pct_team_a",
    "venue_advantage_team_a",
    # Venue behavioral features (from cricmetric)
    "avg_first_innings_score",
    "chase_win_pct",
    "spin_wicket_pct",
    "pace_wicket_pct",
    "boundary_rate",
    "spin_economy",
    "pace_economy",
    # Match condition features
    "dew_probability",
    "pitch_batting_bias",
    "spin_effectiveness",
    "night_match",
]


def build_match_feature_frame(records: list[dict] | pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(records).copy()

    # Inject venue behavioral features if not already present
    venue_numeric = [
        "avg_first_innings_score", "chase_win_pct", "spin_wicket_pct",
        "pace_wicket_pct", "boundary_rate", "spin_economy", "pace_economy",
    ]
    missing = [c for c in venue_numeric if c not in frame.columns]
    if missing and "venue" in frame.columns:
        venue_feats = frame["venue"].map(get_venue_features)
        venue_df = pd.DataFrame(venue_feats.tolist(), index=frame.index)
        for col in missing:
            frame[col] = venue_df[col]

    # Fill dew / pitch / weather defaults if absent
    if "dew_probability" not in frame.columns:
        frame["dew_probability"] = 0.3
    if "pitch_batting_bias" not in frame.columns:
        frame["pitch_batting_bias"] = 0.0
    if "spin_effectiveness" not in frame.columns:
        # Reduce by dew: dew suppresses spin
        frame["spin_effectiveness"] = (1.0 - frame["dew_probability"] * 0.5).clip(0.0, 1.0)
    if "night_match" not in frame.columns:
        frame["night_match"] = 1.0
    else:
        frame["night_match"] = frame["night_match"].astype(float)

    return frame[MATCH_FEATURE_COLUMNS].copy()
