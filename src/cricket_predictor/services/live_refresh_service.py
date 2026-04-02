from __future__ import annotations

from typing import Any

from cricket_predictor.providers.base import LiveDataProvider


class LiveRefreshService:
    def __init__(self, provider: LiveDataProvider) -> None:
        self._provider = provider
        self._latest_matches: list[dict[str, Any]] = []

    async def refresh(self) -> list[dict[str, Any]]:
        self._latest_matches = await self._provider.fetch_live_match_context()
        return self._latest_matches

    def get_latest_matches(self) -> list[dict[str, Any]]:
        return self._latest_matches
