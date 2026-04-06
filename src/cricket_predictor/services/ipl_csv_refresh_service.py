from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from cricket_predictor.config.settings import Settings, get_settings

log = logging.getLogger(__name__)


class IplCsvRefreshService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def refresh_once(self) -> dict[str, int | bool]:
        """Pull the latest IPL CSV data, then refresh predictions/results."""
        if self._settings.ipl_csv_refresh_command:
            await self._run_refresh_command(self._settings.ipl_csv_refresh_command)

        from cricket_predictor.services.prediction_service import get_prediction_service
        from cricket_predictor.services.prediction_tracker import get_prediction_tracker

        tracker = get_prediction_tracker()
        summary = await asyncio.to_thread(tracker.check_results_and_learn)
        predictions = await asyncio.to_thread(tracker.rebuild_upcoming_predictions)

        live_matches = 0
        if self._settings.enable_live_updates and self._settings.live_provider == "ipl_csv":
            live_matches = len(await get_prediction_service().refresh_live_predictions())

        result = {
            "updated": int(summary.get("updated", 0)),
            "retrained": bool(summary.get("retrained", False)),
            "predictions": len(predictions),
            "live_matches": live_matches,
        }
        log.info("IPL CSV refresh completed: %s", result)
        return result

    async def _run_refresh_command(self, command: str) -> None:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            message = (stderr or stdout or b"").decode().strip()
            raise RuntimeError(message or f"command exited with code {process.returncode}")

        if stdout:
            log.info("IPL CSV refresh command output: %s", stdout.decode().strip())
        if stderr:
            log.warning("IPL CSV refresh command stderr: %s", stderr.decode().strip())


@lru_cache
def get_ipl_csv_refresh_service(settings: Settings | None = None) -> IplCsvRefreshService:
    return IplCsvRefreshService(settings or get_settings())