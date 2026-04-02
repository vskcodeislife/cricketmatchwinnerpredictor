import asyncio

from fastapi import APIRouter, Depends

from cricket_predictor.api.schemas import (
    AutoMatchPredictionRequest,
    MatchPredictionRequest,
    PlayerPredictionRequest,
)
from cricket_predictor.config.settings import get_settings
from cricket_predictor.providers.cricinfo_standings import resolve_team_name, venue_advantage
from cricket_predictor.services.prediction_service import PredictionService, get_prediction_service
from cricket_predictor.services.standings_service import StandingsService, get_standings_service

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("/match")
def predict_match(
    payload: MatchPredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> dict:
    return service.predict_match(payload)


@router.post("/match/auto", summary="Predict match — form fetched automatically from live standings")
def predict_match_auto(
    payload: AutoMatchPredictionRequest,
    pred_service: PredictionService = Depends(get_prediction_service),
    standings: StandingsService = Depends(get_standings_service),
) -> dict:
    """Resolve team names via aliases, pull current form from the live standings
    cache, compute venue advantage automatically, then run the prediction."""
    team_a = resolve_team_name(payload.team_a)
    team_b = resolve_team_name(payload.team_b)
    toss_winner = resolve_team_name(payload.toss_winner)

    va = venue_advantage(payload.venue, team_a, team_b)

    full_payload = MatchPredictionRequest(
        team_a=team_a,
        team_b=team_b,
        venue=payload.venue,
        match_format=payload.match_format,
        pitch_type=payload.pitch_type,
        toss_winner=toss_winner,
        toss_decision=payload.toss_decision,
        team_a_recent_form=standings.recent_form(team_a),
        team_b_recent_form=standings.recent_form(team_b),
        team_a_batting_strength=standings.batting_strength(team_a),
        team_b_batting_strength=standings.batting_strength(team_b),
        team_a_bowling_strength=standings.bowling_strength(team_a),
        team_b_bowling_strength=standings.bowling_strength(team_b),
        head_to_head_win_pct_team_a=payload.head_to_head_win_pct_team_a,
        venue_advantage_team_a=va,
        dew_probability=payload.dew_probability,
        pitch_batting_bias=payload.pitch_batting_bias,
        night_match=payload.night_match,
    )

    result = pred_service.predict_match(full_payload)

    # Annotate response with the live stats that were used
    ta_standing = standings.get_team(team_a)
    tb_standing = standings.get_team(team_b)
    result["standings_used"] = {
        team_a: {
            "played": ta_standing.played if ta_standing else "N/A",
            "won": ta_standing.won if ta_standing else "N/A",
            "recent_form": ta_standing.recent_form_str if ta_standing else "N/A",
            "nrr": ta_standing.nrr if ta_standing else "N/A",
            "recent_form_pct": ta_standing.recent_form_pct if ta_standing else 0.5,
        },
        team_b: {
            "played": tb_standing.played if tb_standing else "N/A",
            "won": tb_standing.won if tb_standing else "N/A",
            "recent_form": tb_standing.recent_form_str if tb_standing else "N/A",
            "nrr": tb_standing.nrr if tb_standing else "N/A",
            "recent_form_pct": tb_standing.recent_form_pct if tb_standing else 0.5,
        },
    }
    result["standings_fetched_at"] = standings.fetched_at
    return result


@router.post("/player")
def predict_player(
    payload: PlayerPredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> dict:
    return service.predict_player(payload)


@router.post("/live/refresh")
async def refresh_live_predictions(
    service: PredictionService = Depends(get_prediction_service),
) -> dict:
    return {"matches": await service.refresh_live_predictions()}


@router.get("/live/matches")
def get_live_predictions(
    service: PredictionService = Depends(get_prediction_service),
) -> dict:
    return {"matches": service.get_live_predictions()}


@router.get("/data/status")
def data_status() -> dict:
    """Return cricsheet download metadata: last known sizes and update timestamps."""
    from cricket_predictor.data.cricsheet_loader import CricsheetLoader

    settings = get_settings()
    loader = CricsheetLoader(settings.cricsheet_data_dir)
    return {
        "cricsheet_updates_enabled": settings.enable_cricsheet_updates,
        "check_interval_hours": settings.cricsheet_check_interval_hours,
        "tracked_sources": loader.get_meta(),
    }


@router.post("/data/refresh")
async def trigger_data_refresh(
    service: PredictionService = Depends(get_prediction_service),
) -> dict:
    """Manually trigger a cricsheet size check; download and retrain if data changed."""
    from cricket_predictor.services.data_update_service import DataUpdateService

    settings = get_settings()
    retrained = await asyncio.to_thread(DataUpdateService(settings).check_and_retrain)
    if retrained:
        service.reload_models()
    return {
        "retrained": retrained,
        "message": "Models reloaded from fresh cricsheet data." if retrained else "No new data detected.",
    }
