"""SQLite store for IPL match predictions and their actual outcomes.

Schema
------
match_predictions
    Stores one row per prediction made.  When a result is known the
    ``actual_winner`` and ``is_correct`` columns are filled in.

model_accuracy
    Running tally (updated every time a result is recorded).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator


_DDL = """
CREATE TABLE IF NOT EXISTS match_predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id            TEXT    NOT NULL UNIQUE,
    team_a              TEXT    NOT NULL,
    team_b              TEXT    NOT NULL,
    venue               TEXT    NOT NULL,
    match_date          TEXT    NOT NULL,
    predicted_winner    TEXT    NOT NULL,
    team_a_probability  REAL    NOT NULL,
    team_b_probability  REAL    NOT NULL,
    confidence_score    REAL    NOT NULL,
    explanation         TEXT,
    ai_analysis         TEXT,
    actual_winner       TEXT,
    is_correct          INTEGER,          -- NULL=pending, 1=correct, 0=wrong
    created_at          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS model_accuracy (
    id                          INTEGER PRIMARY KEY CHECK (id = 1),
    total_predictions           INTEGER NOT NULL DEFAULT 0,
    correct_predictions         INTEGER NOT NULL DEFAULT 0,
    wrong_since_last_retrain    INTEGER NOT NULL DEFAULT 0,
    last_retrain_at             TEXT,
    updated_at                  TEXT    NOT NULL
);

INSERT OR IGNORE INTO model_accuracy (id, total_predictions, correct_predictions,
    wrong_since_last_retrain, updated_at)
VALUES (1, 0, 0, 0, datetime('now'));

CREATE TABLE IF NOT EXISTS match_overrides (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id    TEXT,           -- NULL = applies to all upcoming matches
    note        TEXT NOT NULL,  -- raw user text
    parsed_json TEXT NOT NULL,  -- JSON of {team, player, adjustment_type, factor}
    created_at  TEXT NOT NULL
);
"""


class PredictionsDB:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_DDL)
            # Migrate existing DBs that lack ai_analysis column
            cols = {r[1] for r in conn.execute("PRAGMA table_info(match_predictions)").fetchall()}
            if "ai_analysis" not in cols:
                conn.execute("ALTER TABLE match_predictions ADD COLUMN ai_analysis TEXT")

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Predictions CRUD
    # ------------------------------------------------------------------

    def save_prediction(
        self,
        match_id: str,
        team_a: str,
        team_b: str,
        venue: str,
        match_date: str,
        predicted_winner: str,
        team_a_probability: float,
        team_b_probability: float,
        confidence_score: float,
        explanation: str = "",
        ai_analysis: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO match_predictions
                    (match_id, team_a, team_b, venue, match_date, predicted_winner,
                     team_a_probability, team_b_probability, confidence_score,
                     explanation, ai_analysis, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id, team_a, team_b, venue, match_date,
                    predicted_winner, team_a_probability, team_b_probability,
                    confidence_score, explanation, ai_analysis,
                    datetime.now(tz=timezone.utc).isoformat(),
                ),
            )

    def record_result(self, match_id: str, actual_winner: str) -> bool | None:
        """Fill in actual result. Returns True/False/None (correct/wrong/not found)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT predicted_winner, is_correct FROM match_predictions WHERE match_id = ?",
                (match_id,),
            ).fetchone()
            if row is None or row["is_correct"] is not None:
                return None
            is_correct = int(row["predicted_winner"] == actual_winner)
            conn.execute(
                """UPDATE match_predictions
                   SET actual_winner = ?, is_correct = ?
                   WHERE match_id = ?""",
                (actual_winner, is_correct, match_id),
            )
            # Update running tally
            if is_correct:
                conn.execute(
                    """UPDATE model_accuracy SET
                       total_predictions = total_predictions + 1,
                       correct_predictions = correct_predictions + 1,
                       updated_at = datetime('now')
                       WHERE id = 1""",
                )
            else:
                conn.execute(
                    """UPDATE model_accuracy SET
                       total_predictions = total_predictions + 1,
                       wrong_since_last_retrain = wrong_since_last_retrain + 1,
                       updated_at = datetime('now')
                       WHERE id = 1""",
                )
            return bool(is_correct)

    def mark_retrained(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE model_accuracy SET
                   wrong_since_last_retrain = 0,
                   last_retrain_at = datetime('now'),
                   updated_at = datetime('now')
                   WHERE id = 1""",
            )

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_prediction(self, match_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM match_predictions WHERE match_id = ?", (match_id,)
            ).fetchone()
            return dict(row) if row else None

    def update_prediction_analysis(self, match_id: str, ai_analysis: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE match_predictions SET ai_analysis = ? WHERE match_id = ?",
                (ai_analysis, match_id),
            )

    def get_recent_predictions(self, limit: int = 10) -> list[dict]:
        """Return predictions for matches before today (completed dates), most recent first."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM match_predictions
                   WHERE match_date < date('now')
                   ORDER BY match_date DESC, created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_upcoming_predictions(self, limit: int = 7) -> list[dict]:
        """Return saved predictions for future matches (today and beyond)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM match_predictions
                   WHERE match_date >= date('now')
                   ORDER BY match_date ASC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Match overrides (injury notes, pitch reports, etc.)
    # ------------------------------------------------------------------

    def save_override(self, note: str, parsed: dict, match_id: str | None = None) -> int:
        import json as _json
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO match_overrides (match_id, note, parsed_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (match_id, note.strip(), _json.dumps(parsed), now),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_active_overrides(self) -> list[dict]:
        """Return all overrides ordered newest-first."""
        import json as _json
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM match_overrides ORDER BY created_at DESC"
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["parsed"] = _json.loads(d["parsed_json"])
            result.append(d)
        return result

    def delete_override(self, override_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM match_overrides WHERE id = ?", (override_id,))

    def get_next_unpredicted_match(self, scheduled_matches: list[dict]) -> dict | None:
        """Return the first scheduled match that has no prediction yet."""
        with self._connect() as conn:
            predicted_ids = {
                r["match_id"]
                for r in conn.execute("SELECT match_id FROM match_predictions").fetchall()
            }
        for m in scheduled_matches:
            if m["match_id"] not in predicted_ids:
                return m
        return None

    def get_accuracy_stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM model_accuracy WHERE id = 1").fetchone()
            if row is None:
                return {"total": 0, "correct": 0, "accuracy_pct": 0.0, "wrong_since_retrain": 0}
            total = row["total_predictions"]
            correct = row["correct_predictions"]
            return {
                "total": total,
                "correct": correct,
                "accuracy_pct": round(100.0 * correct / total, 1) if total else 0.0,
                "wrong_since_retrain": row["wrong_since_last_retrain"],
                "last_retrain_at": row["last_retrain_at"],
            }

    def get_pending_result_match_ids(self) -> list[str]:
        """Match IDs where prediction is saved but result not yet recorded."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT match_id FROM match_predictions WHERE is_correct IS NULL"
            ).fetchall()
            return [r["match_id"] for r in rows]
