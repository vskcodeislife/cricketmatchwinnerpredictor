# Cricket Predictor Application

This project provides a FastAPI backend for two prediction tasks:

- match winner probability
- player runs expectation

The code starts with synthetic datasets and scikit-learn pipelines so the system is runnable immediately, while keeping the architecture ready for real feeds and richer data sources later.

## Architecture

- `src/cricket_predictor/api`: FastAPI app, routers, and request schemas
- `src/cricket_predictor/data`: synthetic dataset generation and future repository connectors
- `src/cricket_predictor/features`: reusable feature engineering helpers
- `src/cricket_predictor/models`: training and artifact loading
- `src/cricket_predictor/services`: inference orchestration and live refresh logic
- `src/cricket_predictor/providers`: pluggable live feed providers for Cricbuzz/Cricinfo-style sources or proxy APIs
- `data/synthetic`: generated CSVs (`teams.csv`, `players.csv`, `matches.csv`, `venues.csv`)
- `artifacts/models`: trained model artifacts
- `scripts`: operational scripts such as model training
- `tests`: API smoke tests

## Prediction Logic

### Match winner

Inputs include venue, pitch, toss, team form, batting strength, bowling strength, format, and head-to-head context. The training module supports a baseline logistic regression path and currently defaults to a tree-based classifier for improved non-linear handling.

Response payload includes:

- predicted winner
- winning probability by team
- confidence score
- top contributing factors
- explanation string

### Player runs

Inputs include batting position, career average, strike rate, recent form, opponent bowling strength, venue batting average, pitch, and format. The module supports a baseline linear regression path and currently defaults to a random forest regressor.

Response payload includes:

- predicted runs
- min/max range
- confidence score
- top contributing factors
- explanation string

## Live Feed Strategy

The app is designed so live data comes through provider adapters rather than directly coupling to a single website.

- `MockLiveDataProvider` is the default safe bootstrap path.
- `HttpLiveDataProvider` can connect to an upstream service that exposes normalized live cricket match JSON.
- Provider selection is controlled with environment variables.
- Background refresh can be enabled so live predictions keep updating at a configured interval.

Live update endpoints:

- `POST /predict/live/refresh`: pull fresh live contexts from the selected provider and recompute match predictions
- `GET /predict/live/matches`: return the latest cached live predictions

Recommended production approach:

1. Integrate a licensed cricket data API or an internal ingestion service that normalizes live score and ball-by-ball data.
2. Map that payload into the request features used by the prediction service.
3. Run periodic refresh jobs using the configured interval.
4. Treat Cricbuzz/Cricinfo as upstream content sources only through a compliant ingestion layer or licensed data provider rather than coupling the app directly to brittle page scraping.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python scripts/train_models.py
uvicorn cricket_predictor.api.app:app --reload --app-dir src
```

## Environment Variables

- `CRICKET_PREDICTOR_MODEL_ARTIFACT_DIR=artifacts/models`
- `CRICKET_PREDICTOR_SYNTHETIC_DATA_DIR=data/synthetic`
- `CRICKET_PREDICTOR_LIVE_PROVIDER=mock`
- `CRICKET_PREDICTOR_ENABLE_LIVE_UPDATES=false`
- `CRICKET_PREDICTOR_LIVE_REFRESH_SECONDS=60`
- `CRICKET_PREDICTOR_LIVE_PROVIDER_BASE_URL=https://your-normalized-live-feed`
- `CRICKET_PREDICTOR_AZURE_OPENAI_API_KEY=your-key` — Enables AI pre-match analysis via Azure OpenAI GPT-4.1
- `CRICKET_PREDICTOR_AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com`
- `CRICKET_PREDICTOR_AZURE_OPENAI_DEPLOYMENT=gpt-4.1`

## Example Requests

### `POST /predict/match`

```json
{
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
  "venue_advantage_team_a": 1.0
}
```

### `POST /predict/player`

```json
{
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
  "venue_batting_average": 52.0
}
```

## Notes on Real Data

Synthetic data is only the bootstrap layer. To plug in real data later:

- replace synthetic CSV generation with loaders for historical match and player records
- populate provider adapters from a licensed API, data vendor, or internal ingestion pipeline
- retrain periodically and version artifacts in `artifacts/models`

## Testing

```bash
pytest
```
