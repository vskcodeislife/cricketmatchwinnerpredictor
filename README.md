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

Completed match predictions are also fed back into the match-training dataset. Correct calls reinforce the current weighting, and wrong calls add labeled examples that are used on the next self-learning retrain.

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

Local CSV option:

- `IplCsvDataProvider` can read a local IPL CSV export such as the IPL 2026 Kaggle dataset.
- Supported core files are `matches.csv` and `points_table.csv`.
- Optional files are `deliveries.csv` for batting and bowling strength derivation, plus `orange_cap.csv`, `purple_cap.csv`, and `squads.csv` to feed season leader signals such as top run-getters and wicket-takers.
- The same CSV export is also used by the prediction tracker as an extra completed-match result fallback when `CRICKET_PREDICTOR_IPL_CSV_DATA_DIR` is set.
- A daily refresh loop can also run a user-provided sync command first, then reload winners, future predictions, and live contexts from the refreshed CSV files.

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
- `CRICKET_PREDICTOR_IPL_CSV_DATA_DIR=/absolute/path/to/exported/ipl-csv-folder`
- `CRICKET_PREDICTOR_ENABLE_IPL_CSV_REFRESH=true`
- `CRICKET_PREDICTOR_IPL_CSV_REFRESH_HOURS=24`
- `CRICKET_PREDICTOR_IPL_CSV_REFRESH_COMMAND=/absolute/path/to/scripts/refresh_ipl_csv.sh`
- `CRICKET_PREDICTOR_IPL_CSV_DOWNLOAD_URL=https://www.kaggle.com/api/v1/datasets/download/krishd123/ipl-2026-complete-dataset`
- `KAGGLE_USERNAME=your-kaggle-username` (optional)
- `KAGGLE_KEY=your-kaggle-api-key` (optional)
- `CRICKET_PREDICTOR_AZURE_OPENAI_API_KEY=your-key` — Enables AI pre-match analysis via Azure OpenAI GPT-4.1
- `CRICKET_PREDICTOR_AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com`
- `CRICKET_PREDICTOR_AZURE_OPENAI_DEPLOYMENT=gpt-4.1`

For Kaggle-backed updates, the bundled refresh script uses `curl` against the Kaggle dataset download endpoint. If `KAGGLE_USERNAME` and `KAGGLE_KEY` are present it uses them, otherwise it attempts an anonymous download. It then extracts the archive and copies `matches.csv`, `points_table.csv`, `orange_cap.csv`, `purple_cap.csv`, `squads.csv`, and any optional files such as `deliveries.csv` into `CRICKET_PREDICTOR_IPL_CSV_DATA_DIR`.

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

## Model Training & Self-Learning Knowledge Graph

This section is a comprehensive reference for the data lifecycle, training pipeline, retrain triggers, and feature engineering. It can be used as a knowledge graph for future prompting.

### Data Sources

| Source | Type | Location / Config | Files Used | Purpose |
|--------|------|-------------------|------------|---------|
| **Synthetic CSV** | Offline bootstrap | `data/synthetic/` | `matches.csv`, `players.csv`, `teams.csv`, `venues.csv` | Initial training when no real data exists |
| **Cricsheet JSON** | Historical matches | `data/cricsheet/` (`CRICKET_PREDICTOR_CRICSHEET_DATA_DIR`) | `ipl_male_json/*.json`, `t20s_male_json/*.json`, `recently_played_30_male_json/*.json` | Real match parsing for training features + result verification |
| **Kaggle IPL CSV** | Live season data | `CRICKET_PREDICTOR_IPL_CSV_DATA_DIR` | `matches.csv`, `points_table.csv`, `orange_cap.csv`, `purple_cap.csv`, `squads.csv`, `deliveries.csv` (optional) | Team metrics, season leaders, squad rosters, completed-match results |
| **ESPN Cricinfo** | Live standings | `CRICKET_PREDICTOR_CRICINFO_STANDINGS_URL` | HTML page scrape | Points table, NRR, recent form, W/L streaks |
| **Prediction DB** | Internal feedback | `data/predictions.db` (SQLite) | `match_predictions` table | Completed prediction snapshots used as training examples |
| **Injury Report** | External feed | `data/injury_report.json` | JSON file | Player unavailability → override adjustments on bat/bowl strengths |
| **Venue Profiles** | Computed from matches | `data/venue_profiles.json` | JSON file | Venue behavioral features computed from cricsheet ball-by-ball data (avg score, chase win %, spin/pace splits, boundary rate, economy rates) |
| **Squad Profiles** | Static reference | `data/squad_profiles.json` | JSON file | Team composition fallback |

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING ENTRY POINTS                              │
├──────────────┬──────────────────────────────────────────────────────────────┤
│ Manual CLI   │ python scripts/train_models.py [--cricsheet] [--download]   │
│ Self-learn   │ PredictionTrackerService._do_retrain()                      │
│ Cricsheet BG │ DataUpdateService.check_and_retrain() (background loop)     │
│ API endpoint │ POST /predict/data/refresh                                  │
└──────┬───────┴──────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DataUpdateService (Orchestrator)                        │
│                                                                             │
│  1. Parse cricsheet JSON dirs → matches_df, players_df                     │
│  2. Load completed predictions from PredictionsDB                          │
│  3. Backfill leader signals (cap table) on legacy feedback rows            │
│  4. Merge feedback rows into matches_df (dedup by team+date, prefer newer) │
│  5. Validate new model against current on held-out data (quality gate)   │
│  6. Backup current artifacts to *.prev                                   │
│  7. Call train_all(matches_df, players_df)                               │
│  8. Save artifacts to artifacts/models/                                  │
└──────┬──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         training.py: train_all()                           │
│                                                                             │
│  Match Model: GradientBoostingClassifier (sklearn Pipeline)                │
│    preprocessor: StandardScaler (numeric) + OneHotEncoder (categorical)    │
│    target: team_a_win (1/0)                                                │
│                                                                             │
│  Player Model: RandomForestRegressor (sklearn Pipeline)                    │
│    preprocessor: StandardScaler (numeric) + OneHotEncoder (categorical)    │
│    target: weighted combination of career avg, form, SR, position          │
│                                                                             │
│  Output: match_model.joblib, player_model.joblib                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Features

**Match Model — Numeric Features:**

| Feature | Source | Description |
|---------|--------|-------------|
| `team_a_recent_form` | Standings / CSV | Win rate over last 5 matches (0.0–1.0) |
| `team_b_recent_form` | Standings / CSV | Win rate over last 5 matches (0.0–1.0) |
| `team_a_batting_strength` | Standings NRR / CSV | Composite batting rating (0–100) |
| `team_b_batting_strength` | Standings NRR / CSV | Composite batting rating (0–100) |
| `team_a_bowling_strength` | Standings NRR / CSV | Composite bowling rating (0–100) |
| `team_b_bowling_strength` | Standings NRR / CSV | Composite bowling rating (0–100) |
| `head_to_head_win_pct_team_a` | Match history / CSV | H2H win % for team_a over last 7 meetings |
| `venue_advantage_team_a` | Venue mapping | +1.0 (home), -1.0 (away), 0.0 (neutral) |
| `team_a_top_run_getters_runs` | `orange_cap.csv` | Sum of runs by team's top 3 batters this season |
| `team_b_top_run_getters_runs` | `orange_cap.csv` | Sum of runs by team's top 3 batters this season |
| `team_a_top_wicket_takers_wickets` | `purple_cap.csv` | Sum of wickets by team's top 3 bowlers this season |
| `team_b_top_wicket_takers_wickets` | `purple_cap.csv` | Sum of wickets by team's top 3 bowlers this season |
| `avg_first_innings_score` | `venue_profiles.json` (cricsheet) | Historical average 1st innings total at venue |
| `chase_win_pct` | `venue_profiles.json` (cricsheet) | Historical % of chases won at venue |
| `spin_wicket_pct` | `venue_profiles.json` (cricsheet) | % of wickets taken by spin at venue |
| `pace_wicket_pct` | `venue_profiles.json` (cricsheet) | % of wickets taken by pace at venue |
| `boundary_rate` | `venue_profiles.json` (cricsheet) | Boundaries per delivery at venue |
| `spin_economy` | `venue_profiles.json` (cricsheet) | Average spin economy at venue |
| `pace_economy` | `venue_profiles.json` (cricsheet) | Average pace economy at venue |
| `dew_probability` | Request / default | Probability of dew (affects spin, 0.0–1.0) |
| `pitch_batting_bias` | Request / default | Batting-friendly bias (-1.0 to +1.0) |
| `spin_effectiveness` | Derived | `1.0 - dew_probability × 0.5` |
| `night_match` | Request / default | 1.0 for day-night, 0.0 for day match |

**Match Model — Categorical Features:**

| Feature | Values |
|---------|--------|
| `venue` | Stadium name (one-hot encoded) |
| `match_format` | T20, ODI, Test |
| `pitch_type` | batting, bowling, balanced |
| `toss_winner` | Team name |
| `toss_decision` | bat, bowl |

### Retrain Triggers

```
┌─────────────────────────────────────────────────────────────────────┐
│                       RETRAIN TRIGGERS                              │
├───────────────────┬─────────────────────────────────────────────────┤
│                   │                                                 │
│ 1. WRONG STREAK  │ ≥ 5 wrong predictions since last retrain AND    │
│                   │ ≥ 8 total predictions logged                    │
│   Source:         │ PredictionTrackerService.check_results_and_learn│
│   Frequency:      │ Checked every tracker_interval_seconds (1 hr)  │
│                   │                                                 │
│ 2. FEEDBACK       │ ≥ 3 completed predictions since last retrain   │
│    ACCUMULATION   │ (correct or wrong — both reinforce/correct)    │
│   Source:         │ PredictionTrackerService.check_results_and_learn│
│   Frequency:      │ Checked every tracker_interval_seconds (1 hr)  │
│                   │                                                 │
│ 3. CRICSHEET      │ Remote ZIP Content-Length changed (new data)    │
│    DATA UPDATE    │                                                 │
│   Source:         │ DataUpdateService.check_and_retrain()           │
│   Frequency:      │ Every cricsheet_check_interval_hours (24 hr)   │
│   Note:           │ Disabled by default (enable_cricsheet_updates)  │
│                   │                                                 │
│ 4. MANUAL CLI     │ python scripts/train_models.py --cricsheet      │
│                   │                                                 │
│ 5. API ENDPOINT   │ POST /predict/data/refresh                      │
│                   │                                                 │
└───────────────────┴─────────────────────────────────────────────────┘
```

**Validation gate (all triggers):**
Before any retrain commits new artifacts, a quality gate runs:
1. If dataset has < 30 rows, validation is skipped (too noisy to measure)
2. Otherwise, a chronological 80/20 train/validation split is performed
3. A candidate match model is trained on the 80% portion
4. Both the candidate and current deployed model are scored on the 20% held-out set
5. If the candidate accuracy is more than 5% worse than the current model, the retrain is **rejected**
6. If accepted, old artifacts are backed up to `*.prev` before overwriting

### Self-Learning Cycle

```
  ┌──────────────┐     ┌──────────────┐     ┌───────────────┐
  │  IPL Schedule │────▶│   Predict    │────▶│  Save to DB   │
  │  (upcoming)   │     │  (ML model)  │     │ (predictions  │
  └──────────────┘     └──────────────┘     │  .db)         │
                                             └───────┬───────┘
                                                     │
          ┌──────────────────────────────────────────┘
          ▼
  ┌──────────────┐     ┌──────────────┐     ┌───────────────┐
  │ Check Results │────▶│Record Outcome│────▶│ Update Tally  │
  │ (cricsheet,   │     │ (correct /   │     │ (model_       │
  │  CSV, points) │     │  wrong)      │     │  accuracy)    │
  └──────────────┘     └──────────────┘     └───────┬───────┘
                                                     │
          ┌──────────────────────────────────────────┘
          ▼
  ┌──────────────────────────────────────────────────────────┐
  │                   Threshold Check                         │
  │  wrong_since_retrain ≥ 5  OR  completed_since ≥ 3       │
  └────────────────────────┬─────────────────────────────────┘
                           │ YES
                           ▼
  ┌──────────────┐     ┌──────────────┐     ┌───────────────┐
  │  Augment      │────▶│  Retrain     │────▶│  Hot-Reload   │
  │  Training Set │     │  Models      │     │  Models       │
  │  w/ Feedback  │     │  (train_all) │     │  (in-process) │
  └──────────────┘     └──────────────┘     └───────────────┘
```

**Feedback augmentation details:**
- Completed prediction feature snapshots are loaded from SQLite
- Legacy rows missing leader signals get backfilled from current `orange_cap.csv` / `purple_cap.csv`
- Feedback rows are merged with cricsheet match data, deduplicated by `(team_a, team_b, match_date)`, preferring the newer feedback snapshot
- Correct predictions reinforce current weights; wrong predictions add corrective labeled examples

### Background Loops (app.py lifespan)

| Loop | Default Interval | What It Does |
|------|-----------------|--------------|
| **Standings refresh** | 30 min | Scrape ESPN Cricinfo points table → rebuild upcoming predictions |
| **Prediction tracker** | 1 hr | Predict new upcoming matches + check results + trigger retrain if threshold met |
| **Injury refresh** | 12 hr | Fetch injury report → update bat/bowl strength overrides → rebuild predictions |
| **IPL CSV refresh** | 24 hr | Run `refresh_ipl_csv.sh` (Kaggle download) → check results → rebuild predictions → refresh live |
| **Cricsheet update** | 24 hr | HEAD check cricsheet ZIPs → download if changed → retrain (disabled by default) |
| **Live refresh** | 60 sec | Pull live match contexts → recompute in-flight predictions (if enabled) |

### IPL CSV Download Pipeline

```
  ┌────────────────────┐
  │  refresh_ipl_csv.sh│
  │  (cron / BG loop)  │
  └────────┬───────────┘
           │
           ▼
  ┌────────────────────┐     ┌────────────────────┐
  │ curl Kaggle API    │────▶│ Extract ZIP         │
  │ (optional auth)    │     │ (Python zipfile)    │
  └────────────────────┘     └────────┬───────────┘
                                      │
           ┌──────────────────────────┘
           ▼
  ┌────────────────────────────────────────────────┐
  │  Validate required files exist:                │
  │  matches.csv, points_table.csv, orange_cap.csv,│
  │  purple_cap.csv, squads.csv                    │
  └────────┬───────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────────────┐
  │  Copy to CRICKET_PREDICTOR_IPL_CSV_DATA_DIR    │
  └────────┬───────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────────────┐
  │  IplCsvRefreshService.refresh_once()           │
  │  → check_results_and_learn()                   │
  │  → rebuild_upcoming_predictions()              │
  │  → refresh_live_predictions()                  │
  └────────────────────────────────────────────────┘
```

### Result Verification Sources (Priority Order)

When checking if a predicted match has completed, the tracker queries three sources in order. Later sources override earlier ones:

1. **Points table** (`CricinfoStandingsProvider.fetch_recent_results()`) — fastest to update
2. **Local IPL CSV** (`IplCsvDataProvider.fetch_results_lookup()`) — from Kaggle `matches.csv`
3. **Cricsheet JSON** (`recently_played_30_male_json/`) — richest data, preferred when available

### AI Analysis Grounding

Pre-match AI analysis (Azure OpenAI GPT-4.1) is grounded with verified data to prevent hallucinated player names:

- **Squad rosters** from `squads.csv` → injected as `verified_team_a_squad` / `verified_team_b_squad`
- **Batting leaders** from `orange_cap.csv` → `verified_team_a_batting_leaders`
- **Bowling leaders** from `purple_cap.csv` → `verified_team_a_bowling_leaders`
- **Stale detection**: regex scan of existing analysis for player names not in current verified squads → triggers regeneration
- **System prompt**: instructs the LLM to only mention players from the verified lists

### Key Thresholds

| Constant | Value | Location | Effect |
|----------|-------|----------|--------|
| `RETRAIN_WRONG_THRESHOLD` | 5 | `prediction_tracker.py` | Wrong predictions since last retrain to trigger retrain |
| `RETRAIN_MIN_PREDICTIONS` | 8 | `prediction_tracker.py` | Minimum total predictions before wrong-streak retrain fires |
| `RETRAIN_FEEDBACK_THRESHOLD` | 3 | `prediction_tracker.py` | Completed predictions since last retrain to trigger feedback retrain |
| `_MIN_GAMES_FOR_NRR` | 3 | `prediction_tracker.py`, `match_context_service.py` | Minimum team games before using NRR-derived strengths (fallback: 65.0) |

| `_VALIDATION_TOLERANCE` | 0.05 | `data_update_service.py` | Max accuracy drop (5%) tolerated before rejecting a retrain |
| `_MIN_VALIDATION_SAMPLES` | 15 | `data_update_service.py` | Minimum held-out rows needed to run validation (dataset must be ≥ 30) |

### Artifact Storage

| File | Location | Format | Contents |
|------|----------|--------|----------|
| `match_model.joblib` | `artifacts/models/` | sklearn Pipeline | GradientBoostingClassifier with preprocessor |
| `match_model.joblib.prev` | `artifacts/models/` | sklearn Pipeline | Backup of previous match model (auto-created before each retrain) |
| `player_model.joblib` | `artifacts/models/` | sklearn Pipeline | RandomForestRegressor with preprocessor |
| `player_model.joblib.prev` | `artifacts/models/` | sklearn Pipeline | Backup of previous player model (auto-created before each retrain) |
| `predictions.db` | `data/` | SQLite | `match_predictions`, `model_accuracy`, `match_overrides` tables |
| `cricsheet_meta.json` | `data/cricsheet/` | JSON | Content-Length + timestamps per cricsheet URL |

## Notes on Real Data

Synthetic data is only the bootstrap layer. To plug in real data later:

- replace synthetic CSV generation with loaders for historical match and player records
- populate provider adapters from a licensed API, data vendor, or internal ingestion pipeline
- retrain periodically and version artifacts in `artifacts/models`

## Testing

```bash
pytest
```
