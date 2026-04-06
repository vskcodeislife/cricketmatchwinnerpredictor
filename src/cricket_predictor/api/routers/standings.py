import asyncio

from fastapi import APIRouter, Depends

from cricket_predictor.services.prediction_tracker import get_prediction_tracker
from cricket_predictor.services.standings_service import StandingsService, get_standings_service

router = APIRouter(prefix="/standings", tags=["standings"])


@router.get("")
async def get_standings(
    standings: StandingsService = Depends(get_standings_service),
) -> dict:
    """Return the latest cached IPL points table from ESPN Cricinfo."""
    return {
        "fetched_at": standings.fetched_at,
        "table": standings.as_table(),
    }


@router.post("/refresh")
async def refresh_standings(
    standings: StandingsService = Depends(get_standings_service),
) -> dict:
    """Refresh standings and rebuild saved future predictions from the new table."""
    await standings.refresh()
    await asyncio.to_thread(get_prediction_tracker().rebuild_upcoming_predictions)
    return {
        "fetched_at": standings.fetched_at,
        "table": standings.as_table(),
    }
