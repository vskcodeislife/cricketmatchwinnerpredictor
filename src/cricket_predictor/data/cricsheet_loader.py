"""Cricket data loader for cricsheet.org JSON archives.

Checks remote Content-Length headers to detect stale archives, downloads and
extracts ZIP files only when the size has changed, then parses the JSON match
files into DataFrames that are directly compatible with the existing training
pipeline.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

# Use the macOS / system native certificate store when available so that
# Homebrew Python trusts all certificates without extra env-var setup.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import certifi
import httpx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Venue → pitch-type heuristic (lowercase substring matching)
# ---------------------------------------------------------------------------

_VENUE_PITCH_MAP: dict[str, str] = {
    "wankhede": "batting",
    "eden gardens": "batting",
    "chinnaswamy": "batting",
    "chidambaram": "batting",
    "chepauk": "batting",
    "kotla": "batting",
    "jaitley": "batting",
    "rajiv gandhi": "batting",
    "narendra modi": "batting",
    "dubai": "batting",
    "sharjah": "batting",
    "lords": "bowling",
    "lord's": "bowling",
    "headingley": "bowling",
    "edgbaston": "bowling",
    "durban": "bowling",
    "cape town": "bowling",
    "auckland": "bowling",
    "christchurch": "bowling",
    "perth": "bowling",
    "hagley": "bowling",
}

_META_FILENAME = "cricsheet_meta.json"


def _infer_pitch_type(venue: str) -> str:
    lower = venue.lower()
    for keyword, pitch_type in _VENUE_PITCH_MAP.items():
        if keyword in lower:
            return pitch_type
    return "balanced"


def _compute_venue_advantage(venue: str, team_a: str, team_b: str) -> float:
    """Return +1.0 if team_a is the home side, -1.0 if team_b is, else 0.0."""
    from cricket_predictor.providers.cricinfo_standings import venue_advantage
    return venue_advantage(venue, team_a, team_b)


def _normalise_format(raw: str) -> str:
    return {
        "T20I": "T20",
        "IT20": "T20",
        "T20": "T20",
        "ODI": "ODI",
        "Test": "Test",
        "MDM": "Test",
    }.get(raw, "T20")


class CricsheetLoader:
    """Downloads and parses cricsheet.org JSON ZIP archives.

    Archives are only re-downloaded when the remote ``Content-Length`` header
    differs from the locally persisted value, so daily checks are cheap HTTP
    HEAD requests.
    """

    def __init__(self, data_dir: str | Path, timeout_seconds: float = 120.0) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._data_dir / _META_FILENAME
        self._timeout = timeout_seconds
        self._meta: dict[str, dict[str, Any]] = self._load_meta()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_meta(self) -> dict[str, dict[str, Any]]:
        """Return stored metadata (sizes, timestamps) keyed by URL."""
        return dict(self._meta)

    def check_for_updates(self, urls: list[str]) -> bool:
        """Return True if any URL has a different Content-Length than stored."""
        with httpx.Client(timeout=self._timeout, follow_redirects=True, verify=certifi.where()) as client:
            for url in urls:
                try:
                    resp = client.head(url)
                    resp.raise_for_status()
                    remote_size = int(resp.headers.get("content-length", -1))
                    stored_size = self._meta.get(url, {}).get("content_length", -2)
                    if remote_size != stored_size:
                        log.info(
                            "Update detected for %s (remote=%d, stored=%d)",
                            url,
                            remote_size,
                            stored_size,
                        )
                        return True
                except Exception as exc:
                    log.warning("HEAD request failed for %s: %s", url, exc)
        return False

    def download_and_extract(self, urls: list[str]) -> list[Path]:
        """Download changed ZIPs, extract JSON files, and persist metadata.

        Returns a list of directories containing the extracted JSON files.
        Directories whose remote size matches the stored size are included in
        the return list but are not re-downloaded.
        """
        extracted: list[Path] = []
        with httpx.Client(timeout=self._timeout, follow_redirects=True, verify=certifi.where()) as client:
            for url in urls:
                source_name = Path(url).stem  # e.g. "ipl_male_json"
                target_dir = self._data_dir / source_name
                try:
                    head = client.head(url)
                    head.raise_for_status()
                    remote_size = int(head.headers.get("content-length", -1))
                    stored_size = self._meta.get(url, {}).get("content_length", -2)

                    if remote_size == stored_size and target_dir.exists():
                        log.info("No change for %s – skipping download.", url)
                        extracted.append(target_dir)
                        continue

                    log.info("Downloading %s …", url)
                    dl = client.get(url)
                    dl.raise_for_status()
                    target_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(io.BytesIO(dl.content)) as zf:
                        zf.extractall(target_dir)

                    self._meta[url] = {
                        "content_length": remote_size,
                        "last_updated": datetime.utcnow().isoformat(timespec="seconds"),
                        "source_dir": str(target_dir),
                    }
                    self._save_meta()
                    extracted.append(target_dir)
                    log.info("Extracted to %s (%d bytes)", target_dir, remote_size)
                except Exception as exc:
                    log.error("Download/extract failed for %s: %s", url, exc)
        return extracted

    def parse_matches(self, json_dirs: list[Path]) -> pd.DataFrame:
        """Parse all match JSON files and return a training-ready DataFrame.

        Output columns match ``MATCH_FEATURE_COLUMNS`` plus ``team_a`` ,
        ``team_b``, and ``team_a_win``.
        """
        raw: list[dict[str, Any]] = []
        for directory in json_dirs:
            for path in sorted(directory.rglob("*.json")):
                if path.name == _META_FILENAME:
                    continue
                try:
                    record = self._parse_match_file(path)
                    if record is not None:
                        raw.append(record)
                except Exception as exc:
                    log.debug("Skipping match file %s: %s", path.name, exc)

        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw).sort_values("match_date").reset_index(drop=True)
        df = self._compute_rolling_features(df)
        return df

    def parse_player_stats(self, json_dirs: list[Path]) -> pd.DataFrame:
        """Aggregate per-player career stats across all parsed innings.

        Output columns match the player training pipeline schema:
        ``player_name``, ``team``, ``batting_position``, ``career_average``,
        ``strike_rate``, ``recent_form_runs``, ``preferred_format``,
        ``batting_rating``.
        """
        innings: list[dict[str, Any]] = []
        for directory in json_dirs:
            for path in sorted(directory.rglob("*.json")):
                if path.name == _META_FILENAME:
                    continue
                try:
                    innings.extend(self._parse_innings_records(path))
                except Exception as exc:
                    log.debug("Innings parse skipped for %s: %s", path.name, exc)

        if not innings:
            return pd.DataFrame()

        return self._aggregate_player_stats(pd.DataFrame(innings))

    # ------------------------------------------------------------------
    # Match-level parsing
    # ------------------------------------------------------------------

    def _parse_match_file(self, path: Path) -> dict[str, Any] | None:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        info = data.get("info", {})
        teams = info.get("teams", [])
        if len(teams) < 2:
            return None

        team_a, team_b = teams[0], teams[1]
        outcome = info.get("outcome", {})
        winner = outcome.get("winner")
        if not winner or outcome.get("result") in ("no result", "tie"):
            return None

        match_type = _normalise_format(info.get("match_type", "T20"))
        toss = info.get("toss", {})
        toss_winner = toss.get("winner", team_a)
        raw_decision = toss.get("decision", "bat")
        toss_decision = "bowl" if raw_decision == "field" else raw_decision
        venue = info.get("venue") or info.get("city", "Unknown Venue")
        dates = info.get("dates", [])
        match_date = str(dates[0]) if dates else "2000-01-01"

        # Extract per-team run totals and wickets taken from innings blocks
        team_runs: dict[str, int] = {}
        team_wickets_taken: dict[str, int] = {}
        for innings in data.get("innings", []):
            batting_team = innings.get("team")
            if not batting_team:
                continue
            runs = 0
            wickets = 0
            for over in innings.get("overs", []):
                for delivery in over.get("deliveries", []):
                    runs += delivery.get("runs", {}).get("total", 0)
                    wickets += len(delivery.get("wickets", []))
            team_runs[batting_team] = team_runs.get(batting_team, 0) + runs
            bowling_team = team_b if batting_team == team_a else team_a
            team_wickets_taken[bowling_team] = team_wickets_taken.get(bowling_team, 0) + wickets

        return {
            "team_a": team_a,
            "team_b": team_b,
            "venue": venue,
            "match_format": match_type,
            "pitch_type": _infer_pitch_type(venue),
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "match_date": match_date,
            "team_a_win": int(winner == team_a),
            # Temporary columns for rolling feature computation
            "_ta_runs": team_runs.get(team_a, 120),
            "_tb_runs": team_runs.get(team_b, 120),
            "_ta_wkts": team_wickets_taken.get(team_a, 5),
            "_tb_wkts": team_wickets_taken.get(team_b, 5),
        }

    def _compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling form, head-to-head, and strength columns in chronological order."""
        df = df.copy()

        # Mutable rolling state keyed by team name
        recent_wins: dict[str, list[int]] = {}
        team_batting: dict[str, list[float]] = {}
        team_bowling: dict[str, list[float]] = {}
        h2h: dict[tuple[str, str], dict[str, int]] = {}

        computed_rows: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            ta, tb = str(row["team_a"]), str(row["team_b"])
            venue_name = str(row.get("venue", ""))
            key: tuple[str, str] = (min(ta, tb), max(ta, tb))

            # Recent form: win rate over last 5 matches per team
            wta = recent_wins.get(ta, [])[-5:]
            wtb = recent_wins.get(tb, [])[-5:]
            form_a = float(np.mean(wta)) if wta else 0.5
            form_b = float(np.mean(wtb)) if wtb else 0.5

            # Head-to-head win % for team_a
            prev = h2h.get(key, {"wins_a": 0, "total": 0})
            h2h_pct = prev["wins_a"] / prev["total"] if prev["total"] > 0 else 0.5

            # Batting strength: normalise rolling average innings runs to [40, 100]
            bat_a = float(np.clip(np.mean(team_batting.get(ta, [120.0])[-10:]) * 0.45, 40, 100))
            bat_b = float(np.clip(np.mean(team_batting.get(tb, [120.0])[-10:]) * 0.45, 40, 100))

            # Bowling strength: normalise rolling average wickets taken to [40, 100]
            bowl_a = float(np.clip(np.mean(team_bowling.get(ta, [6.0])[-10:]) * 9.0, 40, 100))
            bowl_b = float(np.clip(np.mean(team_bowling.get(tb, [6.0])[-10:]) * 9.0, 40, 100))

            computed_rows.append(
                {
                    "team_a_recent_form": round(form_a, 3),
                    "team_b_recent_form": round(form_b, 3),
                    "head_to_head_win_pct_team_a": round(h2h_pct, 3),
                    "team_a_batting_strength": round(bat_a, 2),
                    "team_b_batting_strength": round(bat_b, 2),
                    "team_a_bowling_strength": round(bowl_a, 2),
                    "team_b_bowling_strength": round(bowl_b, 2),
                    # Compute real venue advantage — cricket venues have identifiable home teams.
                    # Correctly rewards home ground familiarity that the raw cricsheet JSON omits.
                    "venue_advantage_team_a": _compute_venue_advantage(venue_name, ta, tb),
                }
            )

            # Update state *after* recording (to avoid lookahead)
            won = int(row["team_a_win"])
            recent_wins.setdefault(ta, []).append(won)
            recent_wins.setdefault(tb, []).append(1 - won)

            h2h_entry = h2h.setdefault(key, {"wins_a": 0, "total": 0})
            h2h_entry["total"] += 1
            # Ensure h2h is always from the perspective of key[0]
            if ta == key[0]:
                h2h_entry["wins_a"] += won
            else:
                h2h_entry["wins_a"] += 1 - won

            team_batting.setdefault(ta, []).append(float(row["_ta_runs"]))
            team_batting.setdefault(tb, []).append(float(row["_tb_runs"]))
            team_bowling.setdefault(ta, []).append(float(row["_ta_wkts"]))
            team_bowling.setdefault(tb, []).append(float(row["_tb_wkts"]))

        computed = pd.DataFrame(computed_rows, index=df.index)
        result = pd.concat([df, computed], axis=1)
        return result.drop(
            columns=["match_date", "_ta_runs", "_tb_runs", "_ta_wkts", "_tb_wkts"],
            errors="ignore",
        )

    # ------------------------------------------------------------------
    # Player-level parsing
    # ------------------------------------------------------------------

    def _parse_innings_records(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        info = data.get("info", {})
        match_type = _normalise_format(info.get("match_type", "T20"))
        venue = info.get("venue") or info.get("city", "Unknown")
        dates = info.get("dates", [])
        match_date = str(dates[0]) if dates else "2000-01-01"

        records: list[dict[str, Any]] = []
        for innings in data.get("innings", []):
            batting_team = innings.get("team", "Unknown")
            batter_runs: dict[str, int] = {}
            batter_balls: dict[str, int] = {}
            batter_order: dict[str, int] = {}
            position = 0

            for over in innings.get("overs", []):
                for delivery in over.get("deliveries", []):
                    batter = delivery.get("batter", "")
                    if not batter:
                        continue
                    if batter not in batter_order:
                        batter_order[batter] = position + 1
                        position += 1
                    batter_runs[batter] = batter_runs.get(batter, 0) + delivery.get("runs", {}).get("batter", 0)
                    batter_balls[batter] = batter_balls.get(batter, 0) + 1

            for batter, runs in batter_runs.items():
                records.append(
                    {
                        "player_name": batter,
                        "team": batting_team,
                        "batting_position": batter_order.get(batter, 11),
                        "runs": runs,
                        "balls": max(1, batter_balls.get(batter, 1)),
                        "match_format": match_type,
                        "venue": venue,
                        "match_date": match_date,
                    }
                )
        return records

    def _aggregate_player_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("match_date").reset_index(drop=True)

        base = (
            df.groupby("player_name")
            .agg(
                team=("team", "last"),
                innings_count=("runs", "count"),
                career_average=("runs", "mean"),
                total_runs=("runs", "sum"),
                total_balls=("balls", "sum"),
                batting_position=(
                    "batting_position",
                    lambda x: int(x.mode().iloc[0]) if not x.mode().empty else 5,
                ),
                preferred_format=(
                    "match_format",
                    lambda x: x.mode().iloc[0] if not x.mode().empty else "T20",
                ),
            )
            .reset_index()
        )

        recent = (
            df.sort_values("match_date")
            .groupby("player_name")
            .tail(10)
            .groupby("player_name")["runs"]
            .mean()
            .reset_index()
            .rename(columns={"runs": "recent_form_runs"})
        )

        agg = base.merge(recent, on="player_name", how="left")
        agg["strike_rate"] = (agg["total_runs"] / agg["total_balls"].clip(lower=1) * 100).round(2)
        agg["career_average"] = agg["career_average"].round(2)
        agg["recent_form_runs"] = agg["recent_form_runs"].fillna(agg["career_average"]).round(2)
        agg["batting_position"] = agg["batting_position"].clip(1, 11)
        agg["batting_rating"] = (agg["career_average"] + agg["strike_rate"] / 4).round(2)

        # Require at least 3 innings to reduce noise
        agg = agg[agg["innings_count"] >= 3].reset_index(drop=True)

        return agg[
            [
                "player_name",
                "team",
                "batting_position",
                "career_average",
                "strike_rate",
                "recent_form_runs",
                "preferred_format",
                "batting_rating",
            ]
        ]

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict[str, dict[str, Any]]:
        if self._meta_path.exists():
            try:
                with self._meta_path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
        return {}

    def _save_meta(self) -> None:
        with self._meta_path.open("w", encoding="utf-8") as fh:
            json.dump(self._meta, fh, indent=2)
