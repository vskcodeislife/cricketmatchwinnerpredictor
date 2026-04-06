from __future__ import annotations

from types import SimpleNamespace

from cricket_predictor.config.settings import Settings
from cricket_predictor.services.ipl_csv_refresh_service import IplCsvRefreshService


def test_refresh_once_rebuilds_predictions_and_live_matches(monkeypatch, tmp_path) -> None:
    settings = Settings(
        model_artifact_dir=str(tmp_path / "artifacts" / "models"),
        ipl_csv_data_dir=str(tmp_path / "ipl_csv"),
        enable_live_updates=True,
        live_provider="ipl_csv",
    )
    service = IplCsvRefreshService(settings)

    tracker = SimpleNamespace(
        check_results_and_learn=lambda: {"updated": 2, "retrained": True},
        rebuild_upcoming_predictions=lambda: [{"match_id": "m1"}, {"match_id": "m2"}],
    )
    prediction_service = SimpleNamespace(
        refresh_live_predictions=lambda: __import__("asyncio").sleep(0, result=[{"id": 1}])
    )

    monkeypatch.setattr(
        "cricket_predictor.services.prediction_tracker.get_prediction_tracker",
        lambda: tracker,
    )
    monkeypatch.setattr(
        "cricket_predictor.services.prediction_service.get_prediction_service",
        lambda: prediction_service,
    )

    result = __import__("asyncio").run(service.refresh_once())

    assert result == {
        "updated": 2,
        "retrained": True,
        "predictions": 2,
        "live_matches": 1,
    }


def test_refresh_once_runs_configured_command(monkeypatch, tmp_path) -> None:
    settings = Settings(
        model_artifact_dir=str(tmp_path / "artifacts" / "models"),
        ipl_csv_data_dir=str(tmp_path / "ipl_csv"),
        ipl_csv_refresh_command="refresh-command",
    )
    service = IplCsvRefreshService(settings)

    tracker = SimpleNamespace(
        check_results_and_learn=lambda: {"updated": 0, "retrained": False},
        rebuild_upcoming_predictions=lambda: [],
    )
    command_calls: list[str] = []

    async def fake_run_command(command: str) -> None:
        command_calls.append(command)

    monkeypatch.setattr(service, "_run_refresh_command", fake_run_command)
    monkeypatch.setattr(
        "cricket_predictor.services.prediction_tracker.get_prediction_tracker",
        lambda: tracker,
    )

    result = __import__("asyncio").run(service.refresh_once())

    assert command_calls == ["refresh-command"]
    assert result["updated"] == 0
    assert result["predictions"] == 0