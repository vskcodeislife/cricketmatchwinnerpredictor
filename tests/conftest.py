"""Global test fixtures — disable live network calls that can block test runs."""

import pytest


@pytest.fixture(autouse=True)
def _disable_iplt20_stats(monkeypatch):
    """Prevent tests from hitting the live iplt20 S3 endpoints."""
    monkeypatch.setattr(
        "cricket_predictor.providers.iplt20_stats_provider.fetch_team_leader_stats",
        lambda *a, **kw: None,
    )
