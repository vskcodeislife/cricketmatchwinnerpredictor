# IPL Copilot — AI-Powered Cricket Match Predictor

> **Live Demo:** [**ipl-copilot.in**](http://ipl-copilot.in) &nbsp;|&nbsp; [ipl-copilot.onrender.com](https://ipl-copilot.onrender.com/)

A production-grade, self-learning cricket prediction system built with **FastAPI**, **scikit-learn**, and **Azure OpenAI GPT-4.1**. The system predicts IPL match outcomes and player performances using real-time data ingestion, automated model retraining, and LLM-grounded pre-match analysis.

---

## Key Architectural Highlights

| Concern | Design Decision |
|---------|----------------|
| **Self-Learning ML** | Models automatically retrain when prediction accuracy degrades — wrong-streak and feedback-accumulation triggers with a quality gate that rejects retrains that regress >5% |
| **Multi-Source Data Fusion** | Cricsheet JSON + Kaggle CSV + ESPN Cricinfo standings + iplt20.com S3 feeds — resolved via priority-based signal merging with graceful fallbacks |
| **Provider Abstraction** | Pluggable `LiveDataProvider` interface — mock, HTTP, and CSV providers swap via config without code changes |
| **LLM Grounding** | GPT-4.1 pre-match analysis is grounded with verified squad rosters and season leaders to prevent hallucinated player names; stale detection triggers auto-regeneration |
| **Background Orchestration** | Six concurrent background loops (standings, predictions, injuries, CSV refresh, Cricsheet sync, live feed) managed via FastAPI lifespan with configurable intervals |
| **Validation Gate** | Chronological train/validation split prevents model degradation — candidate models must pass accuracy threshold before replacing production artifacts |
| **Hot-Reload Models** | Retrained models are swapped in-process without restart; previous artifacts backed up to `*.prev` for instant rollback |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Application                            │
│                                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Routers  │  │  Prediction  │  │  Match       │  │  Data Update  │  │
│  │  (API)    │──│  Service     │──│  Context     │──│  Service      │  │
│  └──────────┘  └──────────────┘  │  Service     │  └───────────────┘  │
│                                   └──────────────┘                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Provider Layer (Pluggable)                     │  │
│  │  Cricsheet │ IPL CSV │ Standings │ Squad │ Schedule │ Live Feed  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   Background Loop Engine                         │  │
│  │  Standings ∘ Tracker ∘ Injuries ∘ CSV Sync ∘ Cricsheet ∘ Live   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐     │
│  │ scikit-learn  │  │  SQLite      │  │  Azure OpenAI GPT-4.1   │     │
│  │ ML Pipeline   │  │  Feedback DB │  │  (Grounded Analysis)    │     │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Self-Learning Cycle

The system continuously improves without manual intervention:

```
  Schedule  ──▶  Predict  ──▶  Store in DB  ──▶  Verify Results
  (upcoming)     (ML model)    (snapshots)       (cricsheet/CSV/standings)
                                                        │
                                                        ▼
                                                  Threshold Met?
                                               (≥5 wrong OR ≥3 completed)
                                                        │ YES
                                                        ▼
                                              Augment Training Set
                                              ──▶ Retrain (with gate)
                                              ──▶ Hot-Reload Models
```

**Quality gate**: Candidate model must not drop >5% accuracy vs. current model on a chronological held-out split before it replaces production artifacts.

---

## ML Models

| Model | Algorithm | Target | Key Features |
|-------|-----------|--------|-------------|
| **Match Winner** | GradientBoostingClassifier | `team_a_win` (binary) | Team form, bat/bowl strength, H2H record, venue profile (chase win %, spin/pace splits, boundary rate), toss, season leaders, dew probability |
| **Player Runs** | RandomForestRegressor | Weighted runs estimate | Career avg, strike rate, recent form, batting position, opponent bowling strength, venue batting avg |

Both models use sklearn Pipelines with `StandardScaler` + `OneHotEncoder` preprocessing, serialized as `.joblib` artifacts with automatic backup on retrain.

---

## Data Sources & Signal Priority

| Signal | Primary Source | Fallback |
|--------|---------------|----------|
| Recent form | Live standings (scraped) | Cricsheet match history |
| Bat/Bowl strength | Standings-derived NRR | Neutral 65.0 |
| Head-to-head | Cricsheet (last 7 meetings) | CSV provider, then 0.5 |
| Season leaders | Orange/Purple Cap CSV | 0.0 |
| Venue behavior | Computed from Cricsheet ball-by-ball | Default profiles |
| Injury impact | Live injury report | No adjustment |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI, Uvicorn, Pydantic |
| **ML** | scikit-learn (GBM, Random Forest), joblib |
| **Data** | Cricsheet JSON, Kaggle CSV, ESPN Cricinfo, iplt20.com S3 |
| **AI Analysis** | Azure OpenAI GPT-4.1 (grounded with verified rosters) |
| **Storage** | SQLite (predictions DB), JSON (venue/squad/injury profiles) |
| **Deployment** | Render (web service + persistent disk) |
| **Language** | Python 3.11+ |

---

## Project Structure

```
src/cricket_predictor/
├── api/              # FastAPI app, routers, request schemas
├── config/           # Pydantic settings (env-driven)
├── data/             # Data generation & repository connectors
├── features/         # Reusable feature engineering
├── models/           # Training pipeline & artifact loading
├── providers/        # Pluggable data providers (cricsheet, CSV, standings, squads, schedule)
└── services/         # Inference orchestration, live refresh, self-learning tracker
```

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
python scripts/train_models.py
uvicorn cricket_predictor.api.app:app --reload --app-dir src
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict/match` | Match winner prediction with probabilities and contributing factors |
| `POST` | `/predict/player` | Player runs prediction with confidence range |
| `GET` | `/predict/live/matches` | Latest cached live predictions |
| `POST` | `/predict/live/refresh` | Force refresh live predictions from provider |
| `POST` | `/predict/data/refresh` | Trigger data sync + retrain |
| `GET` | `/health` | Health check |

---

## Testing

```bash
pytest
```
