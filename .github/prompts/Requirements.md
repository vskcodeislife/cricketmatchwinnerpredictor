🎯 Role
You are a senior AI/ML engineer and full‑stack developer.
Build a Cricket Predictor Application that:

Predicts which team is likely to win a match
Predicts how many runs a selected player might score

Predictions must be based on match conditions, team strength, player form, and venue factors.
Use modern ML practices, clean architecture, and explainable outputs.

🏗️ Architecture

Backend: Python + FastAPI
ML Layer: scikit‑learn (baseline → tree‑based models)
Data: pandas / numpy
Frontend (optional initially): React or simple HTML
Clear separation:

data/
features/
models/
services/
api/




🧠 Prediction Logic
🏏 Match Winner Prediction
Predict winning probability using:

Team vs team historical win %
Venue (home / away / neutral)
Pitch type (batting / bowling / balanced)
Toss winner & decision
Recent team form (last 5 matches)
Team batting & bowling strength
Match format (T20 / ODI / Test)

Model

Start with Logistic Regression
Upgrade to Random Forest / Gradient Boosting

Output

Winning probability (%) per team
Predicted winner
Confidence score
Top contributing factors


🧍 Player Runs Prediction
Predict expected runs for a player using:

Batting position
Career average & strike rate
Recent form (last N innings)
Opponent bowling strength
Pitch & venue history
Match format

Model

Linear Regression (baseline)
Random Forest Regressor / XGBoost‑style logic

Output

Predicted runs
Min–max range
Confidence level
Key influencing factors


📊 Data Strategy

Start with realistic synthetic datasets
CSV structure:

teams.csv
players.csv
matches.csv
venues.csv


Code must be written so real data (Kaggle / Cricinfo) can be plugged in later


🔌 API Endpoints
Implement using FastAPI:
POST /predict/match
POST /predict/player

Each endpoint should:

Validate input (Pydantic)
Return JSON with prediction, confidence, and explanation


🧪 Explainability (Required)
Every prediction must include:

Why this team/player was predicted
Example:

“Batting‑friendly pitch”
“Strong home record”
“Opponent struggles vs spin”




✅ Engineering Expectations

Clean, readable code
Type hints
No hard‑coded logic
Config‑driven parameters
Clear docstrings
Reusable feature engineering


📦 Deliverables

Folder structure
ML pipeline
Sample dataset generator
FastAPI app
Example API requests & responses
README explaining model logic


🚀 How to Proceed

First design folder structure
Then create synthetic data
Then build feature engineering
Then implement models
Finally expose API endpoints

Think step‑by‑step and produce production‑quality code.

✅ Best Follow‑Up Prompt
After pasting this, ask Copilot:

“Start by designing the folder structure and data schema.”