from types import SimpleNamespace

import pandas as pd

from cricket_predictor.config.settings import Settings
from cricket_predictor.data.predictions_db import PredictionsDB, default_predictions_db_path
from cricket_predictor.providers.cricinfo_standings import CricinfoStandingsProvider
from cricket_predictor.services.prediction_tracker import PredictionTrackerService
from cricket_predictor.services.data_update_service import DataUpdateService


def _feature_snapshot(**overrides: object) -> dict:
    snapshot = {
        "team_a": "Chennai Super Kings",
        "team_b": "Mumbai Indians",
        "venue": "MA Chidambaram Stadium",
        "match_format": "T20",
        "pitch_type": "balanced",
        "toss_winner": "Unknown",
        "toss_decision": "bat",
        "team_a_recent_form": 0.81,
        "team_b_recent_form": 0.52,
        "team_a_batting_strength": 77.0,
        "team_b_batting_strength": 64.0,
        "team_a_bowling_strength": 74.0,
        "team_b_bowling_strength": 62.0,
        "head_to_head_win_pct_team_a": 0.58,
        "venue_advantage_team_a": 1.0,
        "dew_probability": 0.3,
        "pitch_batting_bias": 0.0,
        "night_match": True,
    }
    snapshot.update(overrides)
    return snapshot


def _player_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_name": "Player One",
                "team": "Chennai Super Kings",
                "batting_position": 1,
                "career_average": 34.1,
                "strike_rate": 141.2,
                "recent_form_runs": 39.0,
                "preferred_format": "T20",
                "batting_rating": 71.0,
            }
        ]
    )


def test_completed_predictions_become_training_rows(tmp_path) -> None:
    db = PredictionsDB(tmp_path / "predictions.db")
    snapshot = _feature_snapshot()

    db.save_prediction(
        match_id="ipl-1",
        team_a="Chennai Super Kings",
        team_b="Mumbai Indians",
        venue="MA Chidambaram Stadium",
        match_date="2026-04-05",
        predicted_winner="Mumbai Indians",
        team_a_probability=0.42,
        team_b_probability=0.58,
        confidence_score=0.16,
        explanation="test",
        ai_analysis="",
        feature_snapshot=snapshot,
    )
    db.record_result("ipl-1", "Chennai Super Kings")

    rows = db.get_feedback_training_rows()

    assert len(rows) == 1
    assert rows[0]["team_a"] == "Chennai Super Kings"
    assert rows[0]["team_b"] == "Mumbai Indians"
    assert rows[0]["match_date"] == "2026-04-05"
    assert rows[0]["team_a_recent_form"] == snapshot["team_a_recent_form"]
    assert rows[0]["team_a_win"] == 1
    assert db.count_resolved_predictions_since(None) == 1


def test_local_retrain_prefers_prediction_feedback_for_completed_match(monkeypatch, tmp_path) -> None:
    settings = Settings(
        model_artifact_dir=str(tmp_path / "artifacts" / "models"),
        synthetic_data_dir=str(tmp_path / "data" / "synthetic"),
        cricsheet_data_dir=str(tmp_path / "data" / "cricsheet"),
    )
    db = PredictionsDB(default_predictions_db_path(settings.model_artifact_dir))
    snapshot = _feature_snapshot(team_a_recent_form=0.91, team_b_recent_form=0.33)
    db.save_prediction(
        match_id="ipl-2",
        team_a="Chennai Super Kings",
        team_b="Mumbai Indians",
        venue="MA Chidambaram Stadium",
        match_date="2026-04-05",
        predicted_winner="Mumbai Indians",
        team_a_probability=0.39,
        team_b_probability=0.61,
        confidence_score=0.22,
        explanation="test",
        ai_analysis="",
        feature_snapshot=snapshot,
    )
    db.record_result("ipl-2", "Chennai Super Kings")

    service = DataUpdateService(settings)
    historical_matches = pd.DataFrame(
        [
            {
                "team_a": "Chennai Super Kings",
                "team_b": "Mumbai Indians",
                "venue": "MA Chidambaram Stadium",
                "match_format": "T20",
                "pitch_type": "balanced",
                "toss_winner": "Unknown",
                "toss_decision": "bat",
                "match_date": "2026-04-05",
                "team_a_recent_form": 0.45,
                "team_b_recent_form": 0.64,
                "team_a_batting_strength": 66.0,
                "team_b_batting_strength": 72.0,
                "team_a_bowling_strength": 67.0,
                "team_b_bowling_strength": 71.0,
                "head_to_head_win_pct_team_a": 0.48,
                "venue_advantage_team_a": 1.0,
                "team_a_win": 0,
            }
        ]
    )

    monkeypatch.setattr(service, "_get_local_json_dirs", lambda urls: [tmp_path / "data" / "cricsheet"])
    monkeypatch.setattr(service._loader, "parse_matches", lambda _: historical_matches)
    monkeypatch.setattr(service._loader, "parse_player_stats", lambda _: _player_training_frame())

    captured: dict[str, pd.DataFrame] = {}

    def fake_train_all(matches: pd.DataFrame, players: pd.DataFrame) -> SimpleNamespace:
        captured["matches"] = matches.copy()
        captured["players"] = players.copy()
        return SimpleNamespace(match_model="match", player_model="player")

    monkeypatch.setattr("cricket_predictor.services.data_update_service.train_all", fake_train_all)
    monkeypatch.setattr("cricket_predictor.services.data_update_service.save_artifacts", lambda *args: None)

    assert service.retrain_from_local_data() is True

    trained_matches = captured["matches"]
    assert len(trained_matches) == 1
    assert trained_matches.iloc[0]["team_a_recent_form"] == 0.91
    assert trained_matches.iloc[0]["team_b_recent_form"] == 0.33
    assert trained_matches.iloc[0]["team_a_win"] == 1
    assert len(captured["players"]) == 1


def test_local_retrain_backfills_cap_table_signals_for_legacy_feedback(monkeypatch, tmp_path) -> None:
    dataset_dir = tmp_path / "ipl_csv"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"player_name": "Ruturaj Gaikwad", "runs": 310},
            {"player_name": "Shivam Dube", "runs": 220},
            {"player_name": "Rachin Ravindra", "runs": 180},
            {"player_name": "Suryakumar Yadav", "runs": 280},
            {"player_name": "Tilak Varma", "runs": 175},
            {"player_name": "Rohit Sharma", "runs": 165},
        ]
    ).to_csv(dataset_dir / "orange_cap.csv", index=False)
    pd.DataFrame(
        [
            {"player_name": "Matheesha Pathirana", "wickets": 12},
            {"player_name": "Ravindra Jadeja", "wickets": 9},
            {"player_name": "Noor Ahmad", "wickets": 8},
            {"player_name": "Jasprit Bumrah", "wickets": 11},
            {"player_name": "Trent Boult", "wickets": 7},
            {"player_name": "Deepak Chahar", "wickets": 6},
        ]
    ).to_csv(dataset_dir / "purple_cap.csv", index=False)
    pd.DataFrame(
        [
            {"team": "CSK", "player_name": "Ruturaj Gaikwad"},
            {"team": "CSK", "player_name": "Shivam Dube"},
            {"team": "CSK", "player_name": "Rachin Ravindra"},
            {"team": "CSK", "player_name": "Matheesha Pathirana"},
            {"team": "CSK", "player_name": "Ravindra Jadeja"},
            {"team": "CSK", "player_name": "Noor Ahmad"},
            {"team": "MI", "player_name": "Suryakumar Yadav"},
            {"team": "MI", "player_name": "Tilak Varma"},
            {"team": "MI", "player_name": "Rohit Sharma"},
            {"team": "MI", "player_name": "Jasprit Bumrah"},
            {"team": "MI", "player_name": "Trent Boult"},
        ]
    ).to_csv(dataset_dir / "squads.csv", index=False)

    settings = Settings(
        model_artifact_dir=str(tmp_path / "artifacts" / "models"),
        synthetic_data_dir=str(tmp_path / "data" / "synthetic"),
        cricsheet_data_dir=str(tmp_path / "data" / "cricsheet"),
        ipl_csv_data_dir=str(dataset_dir),
    )
    db = PredictionsDB(default_predictions_db_path(settings.model_artifact_dir))
    db.save_prediction(
        match_id="ipl-legacy-caps",
        team_a="Chennai Super Kings",
        team_b="Mumbai Indians",
        venue="MA Chidambaram Stadium",
        match_date="2026-04-05",
        predicted_winner="Chennai Super Kings",
        team_a_probability=0.56,
        team_b_probability=0.44,
        confidence_score=0.12,
        explanation="test",
        ai_analysis="",
        feature_snapshot=_feature_snapshot(),
    )
    db.record_result("ipl-legacy-caps", "Chennai Super Kings")

    service = DataUpdateService(settings)
    monkeypatch.setattr(service, "_get_local_json_dirs", lambda urls: [tmp_path / "data" / "cricsheet"])
    monkeypatch.setattr(service._loader, "parse_matches", lambda _: pd.DataFrame())
    monkeypatch.setattr(service._loader, "parse_player_stats", lambda _: _player_training_frame())

    captured: dict[str, pd.DataFrame] = {}

    def fake_train_all(matches: pd.DataFrame, players: pd.DataFrame) -> SimpleNamespace:
        captured["matches"] = matches.copy()
        captured["players"] = players.copy()
        return SimpleNamespace(match_model="match", player_model="player")

    monkeypatch.setattr("cricket_predictor.services.data_update_service.train_all", fake_train_all)
    monkeypatch.setattr("cricket_predictor.services.data_update_service.save_artifacts", lambda *args: None)

    assert service.retrain_from_local_data() is True

    trained_match = captured["matches"].iloc[0]
    assert trained_match["team_a_top_run_getters_runs"] == 710.0
    assert trained_match["team_b_top_run_getters_runs"] == 620.0
    assert trained_match["team_a_top_wicket_takers_wickets"] == 29.0
    assert trained_match["team_b_top_wicket_takers_wickets"] == 18.0


def test_tracker_retrains_after_enough_completed_predictions(monkeypatch, tmp_path) -> None:
    settings = Settings(model_artifact_dir=str(tmp_path / "artifacts" / "models"))
    tracker = PredictionTrackerService(settings)
    snapshot = _feature_snapshot()
    matches = {
        "m1": {"match_id": "m1", "team_a": "Chennai Super Kings", "team_b": "Mumbai Indians", "match_date": "2026-04-05"},
        "m2": {"match_id": "m2", "team_a": "Royal Challengers Bengaluru", "team_b": "Delhi Capitals", "match_date": "2026-04-06"},
        "m3": {"match_id": "m3", "team_a": "Punjab Kings", "team_b": "Rajasthan Royals", "match_date": "2026-04-07"},
    }

    for match_id, match in matches.items():
        tracker._db.save_prediction(
            match_id=match_id,
            team_a=match["team_a"],
            team_b=match["team_b"],
            venue="Test Venue",
            match_date=match["match_date"],
            predicted_winner=match["team_a"],
            team_a_probability=0.62,
            team_b_probability=0.38,
            confidence_score=0.24,
            explanation="test",
            ai_analysis="",
            feature_snapshot=snapshot | {"team_a": match["team_a"], "team_b": match["team_b"]},
        )

    monkeypatch.setattr(tracker, "_fetch_cricsheet_results", lambda: {
        (match["team_a"], match["team_b"], match["match_date"]): match["team_a"]
        for match in matches.values()
    })
    monkeypatch.setattr(tracker._schedule, "get_match_by_id", lambda match_id: matches.get(match_id))

    retrain_calls = {"count": 0}

    def fake_retrain() -> bool:
        retrain_calls["count"] += 1
        return True

    monkeypatch.setattr(tracker, "_do_retrain", fake_retrain)

    summary = tracker.check_results_and_learn()

    assert summary["updated"] == 3
    assert summary["retrained"] is True
    assert retrain_calls["count"] == 1


def test_points_table_parser_extracts_recent_results_without_duplicates(monkeypatch) -> None:
        provider = CricinfoStandingsProvider("https://example.com/points-table")
        html = """
        <div class="match-item loss">
            <div class="match-item-head">
                <p class="match-meta match-date">Apr 5 2026</p>
            </div>
            <div class="match-item-body">
                <div class="team team-a">
                    <div class="team-name"><p class="name full-name">Sunrisers Hyderabad</p></div>
                </div>
                <div class="team team-b team-won">
                    <div class="team-name"><p class="name full-name">Lucknow Super Giants</p></div>
                </div>
            </div>
        </div>
        <div class="match-item win">
            <div class="match-item-head">
                <p class="match-meta match-date">Apr 5 2026</p>
            </div>
            <div class="match-item-body">
                <div class="team team-a">
                    <div class="team-name"><p class="name full-name">Sunrisers Hyderabad</p></div>
                </div>
                <div class="team team-b team-won">
                    <div class="team-name"><p class="name full-name">Lucknow Super Giants</p></div>
                </div>
            </div>
        </div>
        """

        monkeypatch.setattr(provider, "_fetch_html", lambda: html)

        results = provider.fetch_recent_results()

        assert len(results) == 1
        assert results[0].match_date == "2026-04-05"
        assert results[0].team_a == "Sunrisers Hyderabad"
        assert results[0].team_b == "Lucknow Super Giants"
        assert results[0].winner == "Lucknow Super Giants"


def test_tracker_uses_points_table_results_when_cricsheet_is_missing(monkeypatch, tmp_path) -> None:
        settings = Settings(model_artifact_dir=str(tmp_path / "artifacts" / "models"))
        tracker = PredictionTrackerService(settings)
        match = {
                "match_id": "m-points",
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
                team_a_probability=0.57,
                team_b_probability=0.43,
                confidence_score=0.14,
                explanation="test",
                ai_analysis="",
                feature_snapshot=_feature_snapshot(team_a=match["team_a"], team_b=match["team_b"]),
        )

        monkeypatch.setattr(tracker, "_fetch_cricsheet_results", lambda: {})
        monkeypatch.setattr(
                tracker,
                "_fetch_points_table_results",
                lambda: {
                        (match["team_a"], match["team_b"], match["match_date"]): match["team_b"],
                        (match["team_b"], match["team_a"], match["match_date"]): match["team_b"],
                },
        )
        monkeypatch.setattr(
                tracker._schedule,
                "get_match_by_id",
                lambda match_id: match if match_id == match["match_id"] else None,
        )

        summary = tracker.check_results_and_learn()
        saved = tracker._db.get_prediction(match["match_id"])

        assert summary["updated"] == 1
        assert summary["retrained"] is False
        assert saved is not None
        assert saved["actual_winner"] == "Lucknow Super Giants"


def test_rebuild_upcoming_predictions_replaces_stale_saved_prediction(monkeypatch, tmp_path) -> None:
        settings = Settings(model_artifact_dir=str(tmp_path / "artifacts" / "models"))
        tracker = PredictionTrackerService(settings)
        future_match = {
                "match_id": "future-1",
                "team_a": "Chennai Super Kings",
                "team_b": "Mumbai Indians",
                "venue": "MA Chidambaram Stadium",
                "match_date": "2099-04-05",
        }

        tracker._db.save_prediction(
                match_id=future_match["match_id"],
                team_a=future_match["team_a"],
                team_b=future_match["team_b"],
                venue=future_match["venue"],
                match_date=future_match["match_date"],
                predicted_winner=future_match["team_b"],
                team_a_probability=0.46,
                team_b_probability=0.54,
                confidence_score=0.08,
                explanation="stale",
                ai_analysis="",
                feature_snapshot=_feature_snapshot(
                        team_a=future_match["team_a"],
                        team_b=future_match["team_b"],
                ),
        )

        standing = SimpleNamespace(played=5)
        standings = SimpleNamespace(
                get_team=lambda _: standing,
                recent_form=lambda team: 0.78 if team == future_match["team_a"] else 0.41,
                batting_strength=lambda team: 79.0 if team == future_match["team_a"] else 66.0,
                bowling_strength=lambda team: 75.0 if team == future_match["team_a"] else 63.0,
        )

        class DummyPredictionService:
                def predict_match(self, payload) -> dict:
                        return {
                                "predicted_winner": payload.team_a,
                                "winning_probability": {
                                        payload.team_a: 63.0,
                                        payload.team_b: 37.0,
                                },
                                "confidence_score": 0.26,
                                "explanation": "refreshed",
                        }

        monkeypatch.setattr(
                "cricket_predictor.services.prediction_tracker.get_standings_service",
                lambda: standings,
        )
        monkeypatch.setattr(
                "cricket_predictor.services.prediction_service.get_prediction_service",
                lambda: DummyPredictionService(),
        )
        monkeypatch.setattr(tracker._schedule, "upcoming_matches", lambda: [future_match])

        tracker.rebuild_upcoming_predictions()
        saved = tracker._db.get_prediction(future_match["match_id"])

        assert saved is not None
        assert saved["predicted_winner"] == "Chennai Super Kings"
        assert saved["explanation"] == "refreshed"