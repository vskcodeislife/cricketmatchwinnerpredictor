from fastapi.testclient import TestClient

from cricket_predictor.api.app import app


client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_match_prediction() -> None:
    response = client.post(
        "/predict/match",
        json={
            "team_a": "India",
            "team_b": "Australia",
            "venue": "Mumbai",
            "match_format": "ODI",
            "pitch_type": "batting",
            "toss_winner": "India",
            "toss_decision": "bat",
            "team_a_recent_form": 0.82,
            "team_b_recent_form": 0.71,
            "team_a_batting_strength": 88,
            "team_b_batting_strength": 81,
            "team_a_bowling_strength": 79,
            "team_b_bowling_strength": 76,
            "head_to_head_win_pct_team_a": 0.58,
            "venue_advantage_team_a": 1.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_winner"] in {"India", "Australia"}
    assert "winning_probability" in payload
    assert "top_contributing_factors" in payload


def test_player_prediction() -> None:
    response = client.post(
        "/predict/player",
        json={
            "player_name": "India Player 1",
            "team": "India",
            "opponent_team": "Australia",
            "venue": "Mumbai",
            "match_format": "ODI",
            "pitch_type": "batting",
            "batting_position": 1,
            "career_average": 49.3,
            "strike_rate": 97.1,
            "recent_form_runs": 61.4,
            "opponent_bowling_strength": 74,
            "venue_batting_average": 52.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["player_name"] == "India Player 1"
    assert payload["predicted_runs"] >= 0
    assert payload["range"]["max"] >= payload["range"]["min"]


def test_live_refresh_endpoint() -> None:
    response = client.post("/predict/live/refresh")
    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    first_match = payload["matches"][0]
    assert "match_context" in first_match
    assert "prediction" in first_match


def test_live_matches_endpoint() -> None:
    client.post("/predict/live/refresh")
    response = client.get("/predict/live/matches")
    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
