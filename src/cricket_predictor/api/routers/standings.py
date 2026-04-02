from fastapi import APIRouter, Depends

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
    """Manually pull the latest standings from ESPN Cricinfo and update the cache."""
    await standings.refresh()
    return {
        "fetched_at": standings.fetched_at,
        "table": standings.as_table(),
    }
