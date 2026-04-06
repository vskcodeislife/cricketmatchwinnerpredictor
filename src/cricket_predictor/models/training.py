from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cricket_predictor.features.match_features import build_match_feature_frame
from cricket_predictor.features.player_features import build_player_feature_frame


MATCH_NUMERIC_COLUMNS = [
    "team_a_recent_form",
    "team_b_recent_form",
    "team_a_batting_strength",
    "team_b_batting_strength",
    "team_a_bowling_strength",
    "team_b_bowling_strength",
    "head_to_head_win_pct_team_a",
    "venue_advantage_team_a",
    "team_a_top_run_getters_runs",
    "team_b_top_run_getters_runs",
    "team_a_top_wicket_takers_wickets",
    "team_b_top_wicket_takers_wickets",
    # Venue behavioral features
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
MATCH_CATEGORICAL_COLUMNS = [
    "venue",
    "match_format",
    "pitch_type",
    "toss_winner",
    "toss_decision",
]

PLAYER_NUMERIC_COLUMNS = [
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
PLAYER_CATEGORICAL_COLUMNS = ["venue", "match_format", "pitch_type"]


@dataclass(slots=True)
class TrainedArtifacts:
    match_model: Pipeline
    player_model: Pipeline


def _match_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), MATCH_NUMERIC_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                MATCH_CATEGORICAL_COLUMNS,
            ),
        ]
    )


def _player_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), PLAYER_NUMERIC_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                PLAYER_CATEGORICAL_COLUMNS,
            ),
        ]
    )


def train_match_model(matches: pd.DataFrame, use_tree_model: bool = True) -> Pipeline:
    estimator = GradientBoostingClassifier(random_state=42) if use_tree_model else LogisticRegression(max_iter=1000)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _match_preprocessor()),
            ("estimator", estimator),
        ]
    )
    pipeline.fit(build_match_feature_frame(matches), matches["team_a_win"])
    return pipeline


def train_player_model(players: pd.DataFrame, use_tree_model: bool = True) -> Pipeline:
    training_frame = players.copy()
    training_frame["opponent_bowling_strength"] = 65.0
    training_frame["venue"] = "Generic Venue"
    training_frame["match_format"] = training_frame["preferred_format"]
    training_frame["pitch_type"] = "balanced"
    training_frame["venue_batting_average"] = training_frame["career_average"] * 1.02
    target = (
        0.55 * training_frame["career_average"]
        + 0.3 * training_frame["recent_form_runs"]
        + 0.08 * training_frame["strike_rate"]
        - 0.15 * training_frame["batting_position"]
        - 0.12 * training_frame["opponent_bowling_strength"]
        + 0.25 * training_frame["venue_batting_average"]
    )
    estimator = RandomForestRegressor(n_estimators=250, random_state=42) if use_tree_model else LinearRegression()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _player_preprocessor()),
            ("estimator", estimator),
        ]
    )
    pipeline.fit(build_player_feature_frame(training_frame), target)
    return pipeline


def train_all(matches: pd.DataFrame, players: pd.DataFrame) -> TrainedArtifacts:
    return TrainedArtifacts(
        match_model=train_match_model(matches),
        player_model=train_player_model(players),
    )


def save_artifacts(artifacts: TrainedArtifacts, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.match_model, output_path / "match_model.joblib")
    joblib.dump(artifacts.player_model, output_path / "player_model.joblib")


def load_match_model(model_dir: str | Path) -> Pipeline:
    return joblib.load(Path(model_dir) / "match_model.joblib")


def load_player_model(model_dir: str | Path) -> Pipeline:
    return joblib.load(Path(model_dir) / "player_model.joblib")
