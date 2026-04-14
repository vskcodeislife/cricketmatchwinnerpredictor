import asyncio
import contextlib
import gc
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cricket_predictor.api.routers.health import router as health_router
from cricket_predictor.api.routers.home import router as home_router
from cricket_predictor.api.routers.predict import router as predict_router
from cricket_predictor.api.routers.standings import router as standings_router
from cricket_predictor.config.settings import get_settings
from cricket_predictor.services.prediction_service import get_prediction_service

log = logging.getLogger(__name__)


async def _live_refresh_loop(refresh_seconds: int) -> None:
    service = get_prediction_service()
    while True:
        try:
            await service.refresh_live_predictions()
        except Exception:
            pass
        await asyncio.sleep(refresh_seconds)


async def _cricsheet_update_loop(interval_seconds: int) -> None:
    """Daily background loop: check cricsheet for new data, retrain if changed."""
    from cricket_predictor.services.data_update_service import DataUpdateService

    settings = get_settings()
    while True:
        try:
            retrained = await asyncio.to_thread(DataUpdateService(settings).check_and_retrain)
            if retrained:
                get_prediction_service().reload_models()
                log.info("Models hot-reloaded after cricsheet retrain.")
        except Exception as exc:
            log.warning("Cricsheet update cycle failed: %s", exc)
        await asyncio.sleep(interval_seconds)


async def _prediction_tracker_loop(interval_seconds: int) -> None:
    """Periodically predict upcoming matches and check past results."""
    from cricket_predictor.services.prediction_tracker import get_prediction_tracker

    tracker = get_prediction_tracker()
    while True:
        try:
            tracker.predict_upcoming_matches()
            summary = tracker.check_results_and_learn()
            if summary["updated"] or summary["retrained"]:
                log.info("Tracker cycle: %s", summary)
        except Exception as exc:
            log.warning("Prediction tracker cycle failed: %s", exc)
        await asyncio.sleep(interval_seconds)


async def _injury_refresh_loop(interval_seconds: int) -> None:
    """Daily loop: fetch injury/unavailability report and update overrides."""
    from cricket_predictor.services.prediction_tracker import get_prediction_tracker

    tracker = get_prediction_tracker()
    while True:
        try:
            count = await asyncio.to_thread(tracker.refresh_injury_overrides)
            if count:
                tracker.predict_upcoming_matches()
                log.info("Injury refresh: %d overrides applied, predictions updated.", count)
        except Exception as exc:
            log.warning("Injury refresh cycle failed: %s", exc)
        await asyncio.sleep(interval_seconds)


async def _standings_refresh_loop(interval_seconds: int) -> None:
    """Periodic loop: refresh IPL standings from ESPN Cricinfo."""
    from cricket_predictor.services.prediction_tracker import get_prediction_tracker
    from cricket_predictor.services.standings_service import get_standings_service

    tracker = get_prediction_tracker()
    while True:
        try:
            await get_standings_service().refresh()
            await asyncio.to_thread(tracker.rebuild_upcoming_predictions)
        except Exception as exc:
            log.warning("Standings refresh/rebuild cycle failed: %s", exc)
        await asyncio.sleep(interval_seconds)


async def _ipl_csv_refresh_loop(interval_seconds: int) -> None:
    """Daily loop: pull configured IPL CSV data and refresh dependent state."""
    from cricket_predictor.services.ipl_csv_refresh_service import get_ipl_csv_refresh_service

    service = get_ipl_csv_refresh_service()
    while True:
        try:
            await service.refresh_once()
        except Exception as exc:
            log.warning("IPL CSV refresh cycle failed: %s", exc)
        await asyncio.sleep(interval_seconds)


async def _scheduled_regenerate_loop() -> None:
    """Run regenerate at 7:30 PM IST and 1:00 AM IST daily."""
    from datetime import datetime, timezone, timedelta

    from cricket_predictor.services.prediction_tracker import get_prediction_tracker
    from cricket_predictor.services.standings_service import get_standings_service

    _IST = timezone(timedelta(hours=5, minutes=30))
    # Target times in IST: 19:30 and 01:00
    SCHEDULE_TIMES = [(19, 30), (1, 0)]

    while True:
        now = datetime.now(_IST)
        # Find the next scheduled time
        candidates = []
        for hour, minute in SCHEDULE_TIMES:
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            candidates.append(target)
        next_run = min(candidates)
        wait_seconds = (next_run - now).total_seconds()
        log.info("Scheduled regenerate: next run at %s IST (in %.0f min).", next_run.strftime("%H:%M"), wait_seconds / 60)
        await asyncio.sleep(wait_seconds)

        try:
            log.info("Scheduled regenerate starting.")
            tracker = get_prediction_tracker()
            # Refresh standings
            try:
                await get_standings_service().refresh()
            except Exception:
                pass
            # Refresh injuries
            try:
                tracker.refresh_injury_overrides()
            except Exception:
                pass
            # Check results and learn
            tracker.check_results_and_learn()
            # Rebuild predictions
            tracker._invalidate_future_predictions()
            tracker.predict_upcoming_matches()
            log.info("Scheduled regenerate completed.")
        except Exception as exc:
            log.warning("Scheduled regenerate failed: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    tasks: list[asyncio.Task] = []

    # --- Bind port FIRST, then initialize in background ---
    # Render requires a port within ~5 minutes; defer all heavy I/O.
    async def _deferred_startup() -> None:
        standings_loaded = False

        # Standings — try once on startup so first prediction is accurate
        if settings.enable_standings_refresh:
            try:
                from cricket_predictor.services.standings_service import get_standings_service
                await get_standings_service().refresh()
                standings_loaded = True
                log.info("Initial standings loaded.")
            except Exception as exc:
                log.warning("Initial standings fetch failed (will retry): %s", exc)

        # Prediction tracker — make pre-match predictions + check results
        try:
            from cricket_predictor.services.prediction_tracker import get_prediction_tracker
            tracker = get_prediction_tracker()

            try:
                count = tracker.refresh_injury_overrides()
                log.info("Initial injury report: %d overrides applied.", count)
            except Exception as exc:
                log.warning("Initial injury report fetch failed: %s", exc)

            if standings_loaded:
                await asyncio.to_thread(tracker.rebuild_upcoming_predictions)
                log.info("Upcoming match predictions rebuilt from refreshed standings.")
            else:
                await asyncio.to_thread(tracker.predict_upcoming_matches)
                log.info("Initial upcoming match predictions saved.")
        except Exception as exc:
            log.warning("Initial prediction tracker run failed: %s", exc)

        if settings.enable_ipl_csv_refresh and settings.ipl_csv_data_dir:
            try:
                from cricket_predictor.services.ipl_csv_refresh_service import get_ipl_csv_refresh_service
                await get_ipl_csv_refresh_service().refresh_once()
            except Exception as exc:
                log.warning("Initial IPL CSV refresh failed (will retry): %s", exc)

        if settings.enable_live_updates:
            try:
                await get_prediction_service().refresh_live_predictions()
            except Exception as exc:
                log.warning("Initial live prediction refresh failed: %s", exc)

        gc.collect()
        log.info("Deferred startup complete.")

    # Launch deferred startup as a background task so the port binds immediately
    tasks.append(asyncio.create_task(_deferred_startup()))

    # Schedule recurring background loops
    if settings.enable_standings_refresh:
        tasks.append(
            asyncio.create_task(
                _standings_refresh_loop(settings.standings_refresh_minutes * 60)
            )
        )
    tasks.append(
        asyncio.create_task(_prediction_tracker_loop(settings.tracker_interval_seconds))
    )
    tasks.append(
        asyncio.create_task(_injury_refresh_loop(12 * 3600))
    )
    if settings.enable_ipl_csv_refresh and settings.ipl_csv_data_dir:
        tasks.append(
            asyncio.create_task(_ipl_csv_refresh_loop(settings.ipl_csv_refresh_hours * 3600))
        )
    if settings.enable_live_updates:
        tasks.append(asyncio.create_task(_live_refresh_loop(settings.live_refresh_seconds)))
    if settings.enable_cricsheet_updates:
        tasks.append(
            asyncio.create_task(
                _cricsheet_update_loop(settings.cricsheet_check_interval_hours * 3600)
            )
        )
    tasks.append(asyncio.create_task(_scheduled_regenerate_loop()))

    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="ML-backed cricket predictor with synthetic bootstrap data and live provider support.",
        lifespan=lifespan,
    )
    app.include_router(home_router)
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(standings_router)
    return app


app = create_app()
