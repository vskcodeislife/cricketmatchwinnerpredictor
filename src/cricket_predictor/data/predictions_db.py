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

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Generator

# IST = UTC+5:30
_IST = timezone(timedelta(hours=5, minutes=30))

def _ist_today_iso() -> str:
    """Return today's date in IST as ISO string."""
    return datetime.now(_IST).date().isoformat()


_FEEDBACK_REQUIRED_COLUMNS = {
    "venue",
    "match_format",
    "pitch_type",
    "toss_winner",
    "toss_decision",
    "team_a_recent_form",
    "team_b_recent_form",
    "team_a_batting_strength",
    "team_b_batting_strength",
    "team_a_bowling_strength",
    "team_b_bowling_strength",
    "head_to_head_win_pct_team_a",
    "venue_advantage_team_a",
}


def default_predictions_db_path(model_artifact_dir: str | Path) -> Path:
    return Path(model_artifact_dir).parent.parent / "data" / "predictions.db"


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
    feature_snapshot_json TEXT,
    actual_winner       TEXT,
    is_correct          INTEGER,          -- NULL=pending, 1=correct, 0=wrong
    resolved_at         TEXT,
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
            # Migrate existing DBs that lack newer columns.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(match_predictions)").fetchall()}
            if "ai_analysis" not in cols:
                conn.execute("ALTER TABLE match_predictions ADD COLUMN ai_analysis TEXT")
            if "feature_snapshot_json" not in cols:
                conn.execute("ALTER TABLE match_predictions ADD COLUMN feature_snapshot_json TEXT")
            if "resolved_at" not in cols:
                conn.execute("ALTER TABLE match_predictions ADD COLUMN resolved_at TEXT")

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
        feature_snapshot: dict | None = None,
    ) -> None:
        feature_snapshot_json = json.dumps(feature_snapshot) if feature_snapshot else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO match_predictions
                    (match_id, team_a, team_b, venue, match_date, predicted_winner,
                     team_a_probability, team_b_probability, confidence_score,
                     explanation, ai_analysis, feature_snapshot_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id, team_a, team_b, venue, match_date,
                    predicted_winner, team_a_probability, team_b_probability,
                    confidence_score, explanation, ai_analysis, feature_snapshot_json,
                    datetime.now(tz=timezone.utc).isoformat(),
                ),
            )

    def record_result(self, match_id: str, actual_winner: str) -> bool | None:
        """Fill in actual result. Returns True/False/None (correct/wrong/not found).

        If *actual_winner* is ``"No Result"`` the match is marked as abandoned
        (``is_correct = -1``) and no accuracy counters are changed.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT predicted_winner, is_correct FROM match_predictions WHERE match_id = ?",
                (match_id,),
            ).fetchone()
            if row is None or row["is_correct"] is not None:
                return None

            # Abandoned / No Result → mark with is_correct = -1
            if actual_winner == "No Result":
                conn.execute(
                    """UPDATE match_predictions
                       SET actual_winner = ?, is_correct = -1, resolved_at = datetime('now')
                       WHERE match_id = ?""",
                    (actual_winner, match_id),
                )
                return None

            is_correct = int(row["predicted_winner"] == actual_winner)
            conn.execute(
                """UPDATE match_predictions
                   SET actual_winner = ?, is_correct = ?, resolved_at = datetime('now')
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
        """Return resolved predictions (past dates OR today with result), most recent first."""
        today_ist = _ist_today_iso()
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM match_predictions
                   WHERE match_date < ? OR actual_winner IS NOT NULL
                   ORDER BY match_date DESC, created_at DESC LIMIT ?""",
                (today_ist, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_paginated_predictions(self, page: int = 1, per_page: int = 10) -> tuple[list[dict], int]:
        """Return a page of past/resolved predictions and total count."""
        today_ist = _ist_today_iso()
        offset = (page - 1) * per_page
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM match_predictions WHERE match_date < ? OR actual_winner IS NOT NULL",
                (today_ist,),
            ).fetchone()[0]
            rows = conn.execute(
                """SELECT * FROM match_predictions
                   WHERE match_date < ? OR actual_winner IS NOT NULL
                   ORDER BY match_date DESC, created_at DESC
                   LIMIT ? OFFSET ?""",
                (today_ist, per_page, offset),
            ).fetchall()
            return [dict(r) for r in rows], total

    def get_upcoming_predictions(self, limit: int = 7) -> list[dict]:
        """Return saved predictions for future matches (today and beyond), excluding resolved."""
        today_ist = _ist_today_iso()
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM match_predictions
                   WHERE match_date >= ? AND actual_winner IS NULL
                   ORDER BY match_date ASC LIMIT ?""",
                (today_ist, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_feedback_training_rows(self) -> list[dict]:
        """Return completed prediction rows as supervised match-training examples."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT team_a, team_b, match_date, actual_winner, feature_snapshot_json
                   FROM match_predictions
                   WHERE actual_winner IS NOT NULL AND feature_snapshot_json IS NOT NULL
                   ORDER BY match_date ASC, created_at ASC"""
            ).fetchall()

        feedback_rows: list[dict] = []
        for row in rows:
            actual_winner = row["actual_winner"]
            team_a = row["team_a"]
            team_b = row["team_b"]
            if actual_winner not in {team_a, team_b}:
                continue
            try:
                snapshot = json.loads(row["feature_snapshot_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(snapshot, dict) or not _FEEDBACK_REQUIRED_COLUMNS.issubset(snapshot):
                continue
            feedback_rows.append(
                {
                    **snapshot,
                    "team_a": team_a,
                    "team_b": team_b,
                    "match_date": row["match_date"],
                    "team_a_win": int(actual_winner == team_a),
                }
            )
        return feedback_rows

    def count_resolved_predictions_since(self, resolved_after: str | None) -> int:
        with self._connect() as conn:
            if resolved_after:
                row = conn.execute(
                    """SELECT COUNT(*) AS count
                       FROM match_predictions
                       WHERE resolved_at IS NOT NULL AND resolved_at > ?""",
                    (resolved_after,),
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT COUNT(*) AS count
                       FROM match_predictions
                       WHERE resolved_at IS NOT NULL"""
                ).fetchone()
        return int(row["count"] if row else 0)

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
            # Compute from actual resolved predictions (is_correct 0 or 1).
            # Abandoned (is_correct = -1) and pending (NULL) are excluded.
            agg = conn.execute(
                """SELECT
                       COUNT(*) AS total,
                       SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS correct
                   FROM match_predictions
                   WHERE is_correct IN (0, 1)"""
            ).fetchone()
            total = agg["total"] or 0
            correct = agg["correct"] or 0

            meta = conn.execute("SELECT last_retrain_at FROM model_accuracy WHERE id = 1").fetchone()
            last_retrain = meta["last_retrain_at"] if meta else None

            # Derive "wrong since retrain" from match rows so the number cannot
            # drift from total/correct counters in older DBs.
            if last_retrain:
                wrong_row = conn.execute(
                    """SELECT COUNT(*) AS wrong
                       FROM match_predictions
                       WHERE is_correct = 0 AND resolved_at IS NOT NULL AND resolved_at > ?""",
                    (last_retrain,),
                ).fetchone()
            else:
                wrong_row = conn.execute(
                    """SELECT COUNT(*) AS wrong
                       FROM match_predictions
                       WHERE is_correct = 0"""
                ).fetchone()
            wrong_since = int(wrong_row["wrong"] if wrong_row else 0)

            return {
                "total": total,
                "correct": correct,
                "accuracy_pct": round(100.0 * correct / total, 1) if total else 0.0,
                "wrong_since_retrain": wrong_since,
                "last_retrain_at": last_retrain,
            }

    def get_pending_result_match_ids(self) -> list[str]:
        """Match IDs where prediction is saved but result not yet recorded."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT match_id FROM match_predictions WHERE is_correct IS NULL"
            ).fetchall()
            return [r["match_id"] for r in rows]
