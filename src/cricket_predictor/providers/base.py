from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LiveDataProvider(ABC):
    @abstractmethod
    async def fetch_live_match_context(self) -> list[dict[str, Any]]:
        """Return current live-match snapshots used to refresh predictions."""
