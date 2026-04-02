from __future__ import annotations

from typing import Any

import httpx

from cricket_predictor.providers.base import LiveDataProvider


class HttpLiveDataProvider(LiveDataProvider):
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def fetch_live_match_context(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.get(f"{self._base_url}/matches/live")
            response.raise_for_status()
            payload = response.json()
        if isinstance(payload, list):
            return payload
        return payload.get("matches", [])
