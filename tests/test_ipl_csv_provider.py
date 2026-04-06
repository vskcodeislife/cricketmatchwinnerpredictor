from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from cricket_predictor.config.settings import Settings
from cricket_predictor.providers.ipl_csv_provider import IplCsvDataProvider
from cricket_predictor.services.prediction_tracker import PredictionTrackerService


def _write_dataset_csvs(dataset_dir, *, matches: list[dict], points_table: list[dict], deliveries: list[dict]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matches).to_csv(dataset_dir / "matches.csv", index=False)
    pd.DataFrame(points_table).to_csv(dataset_dir / "points_table.csv", index=False)
    pd.DataFrame(deliveries).to_csv(dataset_dir / "deliveries.csv", index=False)


def test_ipl_csv_provider_builds_live_match_context(tmp_path) -> None:
    dataset_dir = tmp_path / "ipl_csv"
    today = date.today()
    _write_dataset_csvs(
        dataset_dir,
        matches=[
            {
                "match_id": 1,
                "date": (today - timedelta(days=2)).isoformat(),
                "team_1": "CSK",
                "team_2": "MI",
                "venue": "MA Chidambaram Stadium",
                "winner": "CSK",
                "toss_winner": "CSK",
                "toss_decision": "bat",
                "status": "completed",
            },
            {
                "match_id": 2,
                "date": today.isoformat(),
                "team_1": "Chennai Super Kings",
                "team_2": "Mumbai Indians",
                "venue": "MA Chidambaram Stadium",
                "winner": "",
                "toss_winner": "Chennai Super Kings",
                "toss_decision": "bowl",
                "status": "live",
            },
        ],
        points_table=[
            {"team": "CSK", "played": 4, "won": 3, "form": "W W L W"},
            {"team": "MI", "played": 4, "won": 1, "form": "L W L L"},
        ],
        deliveries=[
            {"match_id": 1, "inning": 1, "batting_team": "CSK", "bowling_team": "MI", "total_runs": 190},
            {"match_id": 1, "inning": 2, "batting_team": "MI", "bowling_team": "CSK", "total_runs": 175},
            {"match_id": 2, "inning": 1, "batting_team": "CSK", "bowling_team": "MI", "total_runs": 182},
            {"match_id": 2, "inning": 2, "batting_team": "MI", "bowling_team": "CSK", "total_runs": 168},
        ],
    )

    provider = IplCsvDataProvider(dataset_dir)
    contexts = __import__("asyncio").run(provider.fetch_live_match_context())

    assert len(contexts) == 1
    context = contexts[0]
    assert context["team_a"] == "Chennai Super Kings"
    assert context["team_b"] == "Mumbai Indians"
    assert context["toss_decision"] == "bowl"
    assert context["team_a_recent_form"] == 0.75
    assert context["team_b_recent_form"] == 0.25
    assert context["head_to_head_win_pct_team_a"] == 1.0
    assert context["venue_advantage_team_a"] == 1.0
    assert context["team_a_batting_strength"] > context["team_b_batting_strength"]


def test_ipl_csv_provider_exposes_completed_match_results(tmp_path) -> None:
    dataset_dir = tmp_path / "ipl_csv"
    _write_dataset_csvs(
        dataset_dir,
        matches=[
            {
                "match_id": 11,
                "date": "2026-04-05",
                "team_1": "SRH",
                "team_2": "LSG",
                "venue": "Rajiv Gandhi International Stadium",
                "winner": "LSG",
                "status": "completed",
            }
        ],
        points_table=[],
        deliveries=[],
    )

    results = IplCsvDataProvider(dataset_dir).fetch_results_lookup()

    assert results[("Sunrisers Hyderabad", "Lucknow Super Giants", "2026-04-05")] == "Lucknow Super Giants"
    assert results[("Lucknow Super Giants", "Sunrisers Hyderabad", "2026-04-05")] == "Lucknow Super Giants"


def test_tracker_uses_local_csv_results_when_other_sources_are_missing(monkeypatch, tmp_path) -> None:
    dataset_dir = tmp_path / "ipl_csv"
    _write_dataset_csvs(
        dataset_dir,
        matches=[
            {
                "match_id": 31,
                "date": "2026-04-05",
                "team_1": "SRH",
                "team_2": "LSG",
                "venue": "Rajiv Gandhi International Stadium",
                "winner": "LSG",
                "status": "completed",
            }
        ],
        points_table=[],
        deliveries=[],
    )

    settings = Settings(
        model_artifact_dir=str(tmp_path / "artifacts" / "models"),
        ipl_csv_data_dir=str(dataset_dir),
    )
    tracker = PredictionTrackerService(settings)
    match = {
        "match_id": "csv-result-1",
        "team_a": "Sunrisers Hyderabad",
        "team_b": "Lucknow Super Giants",
        "match_date": "2026-04-05",
    }

    tracker._db.save_prediction(
        match_id=match["match_id"],
        team_a=match["team_a"],
        team_b=match["team_b"],
        venue="Rajiv Gandhi International Stadium",
        match_date=match["match_date"],
        predicted_winner=match["team_a"],
        team_a_probability=0.56,
        team_b_probability=0.44,
        confidence_score=0.12,
        explanation="test",
        ai_analysis="",
        feature_snapshot={
            "team_a": match["team_a"],
            "team_b": match["team_b"],
            "venue": "Rajiv Gandhi International Stadium",
            "match_format": "T20",
            "pitch_type": "balanced",
            "toss_winner": match["team_a"],
            "toss_decision": "bat",
            "team_a_recent_form": 0.5,
            "team_b_recent_form": 0.5,
            "team_a_batting_strength": 65.0,
            "team_b_batting_strength": 65.0,
            "team_a_bowling_strength": 65.0,
            "team_b_bowling_strength": 65.0,
            "head_to_head_win_pct_team_a": 0.5,
            "venue_advantage_team_a": 0.0,
        },
    )

    monkeypatch.setattr(tracker, "_fetch_points_table_results", lambda: {})
    monkeypatch.setattr(tracker, "_fetch_cricsheet_results", lambda: {})
    monkeypatch.setattr(
        tracker._schedule,
        "get_match_by_id",
        lambda match_id: match if match_id == match["match_id"] else None,
    )

    summary = tracker.check_results_and_learn()
    saved = tracker._db.get_prediction(match["match_id"])

    assert summary["updated"] == 1
    assert saved is not None
    assert saved["actual_winner"] == "Lucknow Super Giants"