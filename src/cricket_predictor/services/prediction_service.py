from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from cricket_predictor.api.schemas import MatchPredictionRequest, PlayerPredictionRequest
from cricket_predictor.config.settings import Settings, get_settings
from cricket_predictor.data.dataset_generator import save_synthetic_datasets
from cricket_predictor.features.match_features import build_match_feature_frame
from cricket_predictor.features.player_features import build_player_feature_frame
from cricket_predictor.models.training import (
    load_match_model,
    load_player_model,
    save_artifacts,
    train_all,
)
from cricket_predictor.providers.registry import build_live_provider
from cricket_predictor.services.live_refresh_service import LiveRefreshService

log = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        model_dir = Path(settings.model_artifact_dir)
        if not (model_dir / "match_model.joblib").exists() or not (model_dir / "player_model.joblib").exists():
            datasets = save_synthetic_datasets(settings.synthetic_data_dir)
            save_artifacts(train_all(datasets.matches, datasets.players), settings.model_artifact_dir)
        self._match_model = load_match_model(settings.model_artifact_dir)
        self._player_model = load_player_model(settings.model_artifact_dir)
        self._refresh_service = LiveRefreshService(build_live_provider(settings))
        self._live_predictions: list[dict] = []

    async def refresh_live_data(self) -> list[dict]:
        return await self._refresh_service.refresh()

    async def refresh_live_predictions(self) -> list[dict]:
        live_matches = await self.refresh_live_data()
        refreshed_predictions: list[dict] = []
        for live_match in live_matches:
            request = MatchPredictionRequest.model_validate(live_match)
            prediction = self.predict_match(request)
            refreshed_predictions.append(
                {
                    "match_context": request.model_dump(),
                    "prediction": prediction,
                }
            )
        self._live_predictions = refreshed_predictions
        return refreshed_predictions

    def get_live_predictions(self) -> list[dict]:
        return self._live_predictions

    def predict_match(self, payload: MatchPredictionRequest) -> dict:
        feature_row = build_match_feature_frame([payload.model_dump()])
        team_a_probability = float(self._match_model.predict_proba(feature_row)[0][1])
        team_b_probability = 1.0 - team_a_probability
        predicted_winner = payload.team_a if team_a_probability >= 0.5 else payload.team_b
        confidence = abs(team_a_probability - 0.5) * 2

        contributing_factors = self._match_explanations(payload)
        return {
            "predicted_winner": predicted_winner,
            "winning_probability": {
                payload.team_a: round(team_a_probability * 100, 2),
                payload.team_b: round(team_b_probability * 100, 2),
            },
            "confidence_score": round(confidence, 3),
            "top_contributing_factors": contributing_factors,
            "explanation": "; ".join(contributing_factors),
        }

    def predict_player(self, payload: PlayerPredictionRequest) -> dict:
        feature_row = build_player_feature_frame([payload.model_dump()])
        predicted_runs = float(self._player_model.predict(feature_row)[0])
        confidence = max(0.1, min(0.95, 1.0 - (payload.opponent_bowling_strength / 150)))
        spread = max(6.0, predicted_runs * 0.2)
        lower_bound = max(0.0, predicted_runs - spread)
        upper_bound = predicted_runs + spread
        factors = self._player_explanations(payload)
        return {
            "player_name": payload.player_name,
            "predicted_runs": round(predicted_runs, 2),
            "range": {
                "min": round(lower_bound, 2),
                "max": round(upper_bound, 2),
            },
            "confidence_score": round(confidence, 3),
            "top_contributing_factors": factors,
            "explanation": "; ".join(factors),
        }

    def _match_explanations(self, payload: MatchPredictionRequest) -> list[str]:
        factors: list[str] = []
        if payload.pitch_type == "batting":
            factors.append("Batting-friendly pitch increases top-order scoring upside")
        elif payload.pitch_type == "bowling":
            factors.append("Bowling-friendly pitch rewards stronger bowling attacks")
        if payload.venue_advantage_team_a > 0:
            factors.append(f"{payload.team_a} carries home or venue familiarity advantage")
        if payload.head_to_head_win_pct_team_a > 0.55:
            factors.append(f"{payload.team_a} has the stronger recent head-to-head record")
        form_gap = payload.team_a_recent_form - payload.team_b_recent_form
        if form_gap > 0.08:
            factors.append(f"{payload.team_a} enters with better recent form")
        elif form_gap < -0.08:
            factors.append(f"{payload.team_b} enters with better recent form")
        if payload.toss_winner == payload.team_a and payload.toss_decision == "bat":
            factors.append(f"{payload.team_a} won the toss and chose the more favorable setup")
        if payload.dew_probability > 0.5 and payload.night_match:
            factors.append(f"Heavy dew expected — chasing side favoured (dew_prob={payload.dew_probability:.0%})")
        if payload.pitch_batting_bias > 0.3:
            factors.append("Batting-biased pitch conditions likely to elevate scores")
        elif payload.pitch_batting_bias < -0.3:
            factors.append("Bowling-biased pitch conditions will put a premium on bowling quality")
        return factors[:4] or ["Model relies on balanced contributions across form, strength, toss, and venue"]

    def _player_explanations(self, payload: PlayerPredictionRequest) -> list[str]:
        factors: list[str] = []
        if payload.pitch_type == "batting":
            factors.append("Batting-friendly pitch supports run accumulation")
        if payload.recent_form_runs > payload.career_average:
            factors.append("Recent form is above long-term batting average")
        if payload.batting_position <= 4:
            factors.append("Top-order batting position provides more scoring opportunity")
        if payload.opponent_bowling_strength >= 75:
            factors.append("Strong opposition bowling attack caps the ceiling")
        if payload.venue_batting_average > payload.career_average:
            factors.append("Venue history is better than the player baseline")
        if payload.dew_probability > 0.5 and payload.night_match:
            factors.append(f"Dew expected to assist batting in the second innings (dew_prob={payload.dew_probability:.0%})")
        if payload.pitch_batting_bias > 0.3:
            factors.append("Pitch conditions favour batters today")
        return factors[:4] or ["Prediction is driven by average, strike rate, venue, and matchup balance"]

    def reload_models(self) -> None:
        """Hot-reload model artifacts from disk after a retrain."""
        self._match_model = load_match_model(self._settings.model_artifact_dir)
        self._player_model = load_player_model(self._settings.model_artifact_dir)
        log.info("Prediction models reloaded from %s", self._settings.model_artifact_dir)


@lru_cache
def get_prediction_service() -> PredictionService:
    return PredictionService(get_settings())
