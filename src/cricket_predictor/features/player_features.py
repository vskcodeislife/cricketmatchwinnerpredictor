from __future__ import annotations

import pandas as pd

from cricket_predictor.features.venue_encoder import get_venue_features


PLAYER_FEATURE_COLUMNS = [
    "venue",
    "match_format",
    "pitch_type",
    "batting_position",
    "career_average",
    "strike_rate",
    "recent_form_runs",
    "opponent_bowling_strength",
    "venue_batting_average",
    # Venue behavioral features
    "avg_first_innings_score",
    "spin_wicket_pct",
    "pace_wicket_pct",
    "boundary_rate",
    # Match condition features
    "dew_probability",
    "pitch_batting_bias",
    "spin_effectiveness",
]


def build_player_feature_frame(records: list[dict] | pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(records).copy()

    venue_numeric = ["avg_first_innings_score", "spin_wicket_pct", "pace_wicket_pct", "boundary_rate"]
    missing = [c for c in venue_numeric if c not in frame.columns]
    if missing and "venue" in frame.columns:
        venue_feats = frame["venue"].map(get_venue_features)
        venue_df = pd.DataFrame(venue_feats.tolist(), index=frame.index)
        for col in missing:
            frame[col] = venue_df[col]

    if "dew_probability" not in frame.columns:
        frame["dew_probability"] = 0.3
    if "pitch_batting_bias" not in frame.columns:
        frame["pitch_batting_bias"] = 0.0
    if "spin_effectiveness" not in frame.columns:
        frame["spin_effectiveness"] = (1.0 - frame["dew_probability"] * 0.5).clip(0.0, 1.0)

    return frame[PLAYER_FEATURE_COLUMNS].copy()
