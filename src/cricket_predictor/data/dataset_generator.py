from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TEAMS = [
    "India",
    "Australia",
    "England",
    "South Africa",
    "New Zealand",
    "Pakistan",
    "Sri Lanka",
    "West Indies",
]

VENUES = [
    ("Mumbai", "batting", "India"),
    ("Melbourne", "balanced", "Australia"),
    ("Lord's", "bowling", "England"),
    ("Cape Town", "balanced", "South Africa"),
    ("Auckland", "bowling", "New Zealand"),
    ("Lahore", "batting", "Pakistan"),
]

FORMATS = ["T20", "ODI", "Test"]
TOSS_DECISIONS = ["bat", "bowl"]


@dataclass(slots=True)
class GeneratedDatasets:
    teams: pd.DataFrame
    players: pd.DataFrame
    matches: pd.DataFrame
    venues: pd.DataFrame


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def build_synthetic_datasets(seed: int = 42) -> GeneratedDatasets:
    rng = np.random.default_rng(seed)

    team_rows: list[dict] = []
    for team in TEAMS:
        team_rows.append(
            {
                "team": team,
                "batting_strength": round(rng.uniform(55, 92), 2),
                "bowling_strength": round(rng.uniform(50, 90), 2),
                "fielding_strength": round(rng.uniform(45, 88), 2),
                "recent_form": round(rng.uniform(0.35, 0.9), 3),
            }
        )
    teams = pd.DataFrame(team_rows)

    venue_rows = [
        {"venue": venue, "pitch_type": pitch_type, "home_team": home_team}
        for venue, pitch_type, home_team in VENUES
    ]
    venues = pd.DataFrame(venue_rows)

    player_rows: list[dict] = []
    for team in TEAMS:
        team_strength = teams.loc[teams["team"] == team].iloc[0]
        for batting_position in range(1, 12):
            base_average = max(8, 48 - batting_position * 2.3 + rng.normal(0, 3))
            player_rows.append(
                {
                    "player_name": f"{team} Player {batting_position}",
                    "team": team,
                    "batting_position": batting_position,
                    "career_average": round(base_average, 2),
                    "strike_rate": round(rng.uniform(65, 165), 2),
                    "recent_form_runs": round(max(5, base_average + rng.normal(0, 12)), 2),
                    "preferred_format": rng.choice(FORMATS),
                    "batting_rating": round((team_strength["batting_strength"] + base_average) / 2, 2),
                }
            )
    players = pd.DataFrame(player_rows)

    matches_rows: list[dict] = []
    for _ in range(450):
        team_a, team_b = rng.choice(TEAMS, size=2, replace=False)
        venue, pitch_type, home_team = VENUES[rng.integers(0, len(VENUES))]
        format_name = rng.choice(FORMATS, p=[0.45, 0.35, 0.20])
        team_a_stats = teams.loc[teams["team"] == team_a].iloc[0]
        team_b_stats = teams.loc[teams["team"] == team_b].iloc[0]
        toss_winner = rng.choice([team_a, team_b])
        toss_decision = rng.choice(TOSS_DECISIONS)
        head_to_head = rng.uniform(0.35, 0.65)
        venue_advantage_team_a = 1.0 if home_team == team_a else -1.0 if home_team == team_b else 0.0
        # Match condition features
        night_match = float(rng.integers(0, 2))
        dew_probability = float(rng.uniform(0.1, 0.7)) if night_match else float(rng.uniform(0.0, 0.2))
        pitch_batting_bias = {"batting": 0.5, "balanced": 0.0, "bowling": -0.5}[pitch_type] + float(rng.normal(0, 0.15))
        pitch_batting_bias = float(np.clip(pitch_batting_bias, -1.0, 1.0))
        spin_effectiveness = float(np.clip(1.0 - dew_probability * 0.5, 0.0, 1.0))
        format_factor = {"T20": 0.1, "ODI": 0.0, "Test": -0.05}[format_name]
        pitch_factor = {"batting": 0.08, "balanced": 0.0, "bowling": -0.06}[pitch_type]
        toss_factor = 0.04 if toss_winner == team_a and toss_decision == "bat" else -0.02
        dew_factor = 0.03 if night_match and dew_probability > 0.4 and toss_winner == team_a and toss_decision == "bowl" else 0.0
        logit = (
            0.03 * (team_a_stats["batting_strength"] - team_b_stats["batting_strength"])
            + 0.025 * (team_a_stats["bowling_strength"] - team_b_stats["bowling_strength"])
            + 1.1 * (team_a_stats["recent_form"] - team_b_stats["recent_form"])
            + 0.9 * (head_to_head - 0.5)
            + 0.18 * venue_advantage_team_a
            + format_factor
            + pitch_factor
            + toss_factor
            + dew_factor
        )
        probability_team_a = float(_sigmoid(np.array([logit]))[0])
        matches_rows.append(
            {
                "team_a": team_a,
                "team_b": team_b,
                "venue": venue,
                "match_format": format_name,
                "pitch_type": pitch_type,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "team_a_recent_form": team_a_stats["recent_form"],
                "team_b_recent_form": team_b_stats["recent_form"],
                "team_a_batting_strength": team_a_stats["batting_strength"],
                "team_b_batting_strength": team_b_stats["batting_strength"],
                "team_a_bowling_strength": team_a_stats["bowling_strength"],
                "team_b_bowling_strength": team_b_stats["bowling_strength"],
                "head_to_head_win_pct_team_a": round(head_to_head, 3),
                "venue_advantage_team_a": venue_advantage_team_a,
                "dew_probability": round(dew_probability, 3),
                "pitch_batting_bias": round(pitch_batting_bias, 3),
                "spin_effectiveness": round(spin_effectiveness, 3),
                "night_match": night_match,
                "team_a_win": int(rng.uniform() < probability_team_a),
            }
        )
    matches = pd.DataFrame(matches_rows)

    return GeneratedDatasets(teams=teams, players=players, matches=matches, venues=venues)


def save_synthetic_datasets(output_dir: str | Path, seed: int = 42) -> GeneratedDatasets:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    datasets = build_synthetic_datasets(seed=seed)
    datasets.teams.to_csv(output_path / "teams.csv", index=False)
    datasets.players.to_csv(output_path / "players.csv", index=False)
    datasets.matches.to_csv(output_path / "matches.csv", index=False)
    datasets.venues.to_csv(output_path / "venues.csv", index=False)
    return datasets
