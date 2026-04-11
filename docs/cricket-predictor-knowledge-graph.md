# Cricket Predictor — Knowledge Graph

## Data Sources

| Source | Type | Config / Location | Key Files | Purpose |
|--------|------|-------------------|-----------|---------|
| **Synthetic CSV** | Offline bootstrap | `data/synthetic/` | `matches.csv`, `players.csv`, `teams.csv`, `venues.csv` | Initial training when no real match data exists |
| **Cricsheet JSON** | Historical matches | `data/cricsheet/` | `ipl_male_json/*.json`, `t20s_male_json/*.json`, `recently_played_30_male_json/*.json` | Real match feature extraction + result verification |
| **Kaggle IPL CSV (primary)** | Live season | `CRICKET_PREDICTOR_IPL_CSV_DATA_DIR` | `matches.csv`, `points_table.csv`, `orange_cap.csv`, `purple_cap.csv`, `squads.csv`, `deliveries.csv` | Team metrics, leaders, squads, completed results |
| **Kaggle Ball-by-Ball (alt)** | Live season fallback | `CRICKET_PREDICTOR_IPL_CSV_DOWNLOAD_URL_ALT` | Single ball-by-ball CSV → auto-normalised | Fallback when primary Kaggle source is stale |
| **ESPN Cricinfo** | Live standings | Scraped HTML | Points table page | Points table, NRR, recent form, W/L streaks |
| **Prediction DB** | Internal feedback | `data/predictions.db` (SQLite) | `match_predictions` table | Completed prediction feature snapshots → retraining |
| **Injury Report** | External feed | `data/injury_report.json` | JSON | Player unavailability → bat/bowl strength overrides |
| **Venue Profiles** | Computed from cricsheet | `data/venue_profiles.json` | JSON | Avg score, chase win %, spin/pace splits, boundary rate, economy |
| **Squad Profiles** | Static reference | `data/squad_profiles.json` | JSON | Team composition fallback |

## IPL Standings & Points Table

- Scraped from ESPN Cricinfo every **30 minutes** via `CricinfoStandingsProvider`
- Provides: team wins, losses, points, NRR, recent form (last 5 W/L)
- Used to derive: `batting_strength`, `bowling_strength`, `recent_form` for each team
- NRR-based strengths only used once a team has played ≥1 game (fallback: 65.0)
- Recent results from standings are also used to verify completed match winners

## Top Batsmen & Bowlers (Season Leaders)

- **Orange Cap** (`orange_cap.csv`): Top run scorers this season per team
  - Feature: `team_a_top_run_getters_runs` / `team_b_top_run_getters_runs` (sum of top 3 batters)
- **Purple Cap** (`purple_cap.csv`): Top wicket takers this season per team
  - Feature: `team_a_top_wicket_takers_wickets` / `team_b_top_wicket_takers_wickets` (sum of top 3 bowlers)
- Used in AI analysis grounding: GPT-4.1 only mentions players from verified squad/leader lists
- Refreshed daily via Kaggle CSV download pipeline

## Previous IPL Matches (Head-to-Head)

- Computed from cricsheet JSON: last 7 meetings between two teams
- Feature: `head_to_head_win_pct_team_a` (0.0–1.0)
- CSV provider H2H is used as fallback, then 0.5 (neutral)

## Training Pipeline

```
Entry Points:
  - CLI:        python scripts/train_models.py [--cricsheet] [--download]
  - Self-learn: PredictionTrackerService._do_retrain()
  - Background: DataUpdateService.check_and_retrain()
  - API:        POST /predict/data/refresh

Pipeline (DataUpdateService):
  1. Parse cricsheet JSON → matches_df, players_df
  2. Load completed predictions from SQLite (with feature snapshots)
  3. Backfill leader signals on legacy feedback rows from orange/purple cap CSVs
  4. Merge feedback rows into training set (dedup by team+date, prefer newer)
  5. Train candidate model
  6. Validate against current model (quality gate: ≤5% accuracy drop)
  7. If accepted: backup old → save new → hot-reload in process

Models:
  - Match:  GradientBoostingClassifier (sklearn Pipeline)
  - Player: RandomForestRegressor (sklearn Pipeline)
  - Both use StandardScaler (numeric) + OneHotEncoder (categorical)
```

## Model Features (Match Prediction)

**Numeric (21 features):**
- `team_a/b_recent_form` — win rate over last 5 matches
- `team_a/b_batting_strength` — composite rating from NRR/CSV (0–100)
- `team_a/b_bowling_strength` — composite rating from NRR/CSV (0–100)
- `head_to_head_win_pct_team_a` — H2H win % last 7 meetings
- `venue_advantage_team_a` — +1 home, -1 away, 0 neutral
- `team_a/b_top_run_getters_runs` — sum of top 3 batters' runs
- `team_a/b_top_wicket_takers_wickets` — sum of top 3 bowlers' wickets
- `avg_first_innings_score` — historical venue avg 1st innings total
- `chase_win_pct` — historical chase win % at venue
- `spin/pace_wicket_pct` — wicket distribution by type at venue
- `boundary_rate` — boundaries per delivery at venue
- `spin/pace_economy` — avg economy rates at venue
- `dew_probability`, `pitch_batting_bias`, `spin_effectiveness`, `night_match`

**Categorical (5 features):**
- `venue`, `match_format`, `pitch_type`, `toss_winner`, `toss_decision`

## Self-Learning Cycle

```
Match Completes
  → Hourly tracker picks up result from Cricinfo/CSV/Cricsheet
  → Records actual_winner → marks prediction ✅ correct or ❌ wrong
  → Updates accuracy counters

Retrain Triggers:
  1. ≥5 wrong predictions since last retrain (AND ≥8 total) → retrain
  2. ≥3 completed predictions since last retrain → retrain
  3. Cricsheet ZIP size changed (new data) → retrain (disabled by default)

Retrain Process:
  → Completed predictions (with feature snapshots) merged into training
  → Correct calls reinforce weights; wrong calls add corrective examples
  → Quality gate: reject if >5% accuracy regression
  → Hot-reload model in process (no restart needed)
```

## Signal Priority (Feature Resolution)

| Signal | Primary Source | Fallback |
|--------|---------------|----------|
| recent_form | Current-season standings | Cricsheet multi-season history |
| batting_strength | Standings (if ≥1 game) | Neutral 65.0 |
| bowling_strength | Standings (if ≥1 game) | Neutral 65.0 |
| head_to_head | Cricsheet last 7 | CSV provider, then 0.5 |
| leader_stats | IPL CSV (orange/purple cap) | 0.0 |

**Note:** If IPL CSV data dir is configured, CSV-derived metrics **override** standings for form, batting/bowling strength, and H2H.

## Background Loops

| Loop | Interval | Description |
|------|----------|-------------|
| Standings refresh | 30 min | Scrape Cricinfo → rebuild predictions |
| Prediction tracker | 1 hr | Predict upcoming + check results + retrain if needed |
| Injury refresh | 12 hr | Fetch injury report → update overrides → rebuild |
| IPL CSV refresh | 24 hr | Kaggle download → check results → rebuild + refresh live |
| Cricsheet update | 24 hr | HEAD check ZIPs → download if changed → retrain (disabled) |
| Live refresh | 60 sec | Pull live contexts → recompute in-flight predictions |

## Result Verification (Priority Order)

1. **Points table** (Cricinfo) — fastest to update after match ends
2. **IPL CSV** (Kaggle `matches.csv`) — daily refresh
3. **Cricsheet JSON** (`recently_played_30`) — richest data, preferred when available

## Key Thresholds

| Constant | Value | Effect |
|----------|-------|--------|
| RETRAIN_WRONG_THRESHOLD | 5 | Wrong predictions since retrain to trigger |
| RETRAIN_MIN_PREDICTIONS | 8 | Min total predictions before wrong-streak retrain |
| RETRAIN_FEEDBACK_THRESHOLD | 3 | Completed predictions since retrain to trigger |
| _MIN_GAMES_FOR_NRR | 1 | Min team games for standings-derived strengths |
| _VALIDATION_TOLERANCE | 0.05 | Max accuracy drop tolerated (5%) |
| _MIN_VALIDATION_SAMPLES | 15 | Min held-out rows for validation |

## Deployment

- Hosted on Render (ipl-copilot.onrender.com)
- Persistent disk at `/app/data` (2 GB) stores models, predictions DB, CSV data
- Auto-deploys from `main` branch of GitHub repo
- Start command copies `.prev` model backups if no current models exist

## AI Analysis (GPT-4.1)

- Azure OpenAI GPT-4.1 generates pre-match analysis
- Grounded with verified data: squad rosters, batting/bowling leaders
- Stale detection: scans existing analysis for player names not in current squads → regenerates
- System prompt restricts LLM to only mention players from verified lists
