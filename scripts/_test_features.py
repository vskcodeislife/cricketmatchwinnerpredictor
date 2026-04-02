"""Quick smoke-test for the updated feature pipeline."""
import sys
sys.path.insert(0, "src")

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from cricket_predictor.features.match_features import build_match_feature_frame, MATCH_FEATURE_COLUMNS
from cricket_predictor.features.player_features import build_player_feature_frame, PLAYER_FEATURE_COLUMNS

row = {
    "venue": "Eden Gardens", "match_format": "T20", "pitch_type": "balanced",
    "toss_winner": "KKR", "toss_decision": "bat",
    "team_a_recent_form": 0.6, "team_b_recent_form": 0.5,
    "team_a_batting_strength": 70.0, "team_b_batting_strength": 68.0,
    "team_a_bowling_strength": 65.0, "team_b_bowling_strength": 67.0,
    "head_to_head_win_pct_team_a": 0.55, "venue_advantage_team_a": 1.0,
}
frame = build_match_feature_frame([row])
print("Match columns match spec:", list(frame.columns) == MATCH_FEATURE_COLUMNS)
vals = frame.iloc[0][["avg_first_innings_score", "chase_win_pct", "spin_economy", "dew_probability", "night_match"]]
print("Venue + dew features:", vals.to_dict())

prow = {
    "venue": "Wankhede Stadium", "match_format": "T20", "pitch_type": "batting",
    "batting_position": 3, "career_average": 42.0, "strike_rate": 145.0,
    "recent_form_runs": 48.0, "opponent_bowling_strength": 70.0, "venue_batting_average": 44.0,
}
pframe = build_player_feature_frame([prow])
print("Player columns match spec:", list(pframe.columns) == PLAYER_FEATURE_COLUMNS)
pvals = pframe.iloc[0][["avg_first_innings_score", "spin_wicket_pct", "dew_probability"]]
print("Player venue + dew features:", pvals.to_dict())
print("ALL OK")
