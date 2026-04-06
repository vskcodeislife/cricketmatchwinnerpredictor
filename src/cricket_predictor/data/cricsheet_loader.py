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
        batter_runs: dict[str, dict[str, int]] = {team_a: {}, team_b: {}}
        bowler_wickets: dict[str, dict[str, int]] = {team_a: {}, team_b: {}}
        non_bowler_dismissals = {"run out", "retired hurt", "retired out", "obstructing the field"}
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
                    batter = delivery.get("batter")
                    if batter:
                        batter_runs.setdefault(batting_team, {})[batter] = (
                            batter_runs.setdefault(batting_team, {}).get(batter, 0)
                            + delivery.get("runs", {}).get("batter", 0)
                        )
                    dismissal_list = delivery.get("wickets", [])
                    if not dismissal_list:
                        continue
                    bowler = delivery.get("bowler")
                    bowling_team = team_b if batting_team == team_a else team_a
                    if not bowler:
                        continue
                    credited = sum(
                        1
                        for wicket in dismissal_list
                        if str(wicket.get("kind", "")).lower() not in non_bowler_dismissals
                    )
                    if credited:
                        bowler_wickets.setdefault(bowling_team, {})[bowler] = (
                            bowler_wickets.setdefault(bowling_team, {}).get(bowler, 0) + credited
                        )
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
            "_ta_batter_runs": batter_runs.get(team_a, {}),
            "_tb_batter_runs": batter_runs.get(team_b, {}),
            "_ta_bowler_wickets": bowler_wickets.get(team_a, {}),
            "_tb_bowler_wickets": bowler_wickets.get(team_b, {}),
        }

    def _compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling form, head-to-head, and strength columns in chronological order."""
        df = df.copy()

        # Mutable rolling state keyed by team name
        recent_wins: dict[str, list[int]] = {}
        team_batting: dict[str, list[float]] = {}
        team_bowling: dict[str, list[float]] = {}
        h2h: dict[tuple[str, str], list[int]] = {}
        season_batter_totals: dict[tuple[str, str], dict[str, int]] = {}
        season_bowler_totals: dict[tuple[str, str], dict[str, int]] = {}

        computed_rows: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            ta, tb = str(row["team_a"]), str(row["team_b"])
            venue_name = str(row.get("venue", ""))
            key: tuple[str, str] = (min(ta, tb), max(ta, tb))
            season_key_a = (str(row.get("match_date", "2000-01-01"))[:4], ta)
            season_key_b = (str(row.get("match_date", "2000-01-01"))[:4], tb)

            # Recent form: win rate over last 5 matches per team
            wta = recent_wins.get(ta, [])[-5:]
            wtb = recent_wins.get(tb, [])[-5:]
            form_a = float(np.mean(wta)) if wta else 0.5
            form_b = float(np.mean(wtb)) if wtb else 0.5

            # Head-to-head win % for team_a over the last 7 meetings.
            prev = h2h.get(key, [])[-7:]
            if prev:
                if ta == key[0]:
                    h2h_pct = float(np.mean(prev))
                else:
                    h2h_pct = float(np.mean([1 - outcome for outcome in prev]))
            else:
                h2h_pct = 0.5

            # Batting strength: normalise rolling average innings runs to [40, 100]
            bat_a = float(np.clip(np.mean(team_batting.get(ta, [120.0])[-10:]) * 0.45, 40, 100))
            bat_b = float(np.clip(np.mean(team_batting.get(tb, [120.0])[-10:]) * 0.45, 40, 100))

            # Bowling strength: normalise rolling average wickets taken to [40, 100]
            bowl_a = float(np.clip(np.mean(team_bowling.get(ta, [6.0])[-10:]) * 9.0, 40, 100))
            bowl_b = float(np.clip(np.mean(team_bowling.get(tb, [6.0])[-10:]) * 9.0, 40, 100))

            top_runs_a = sum(sorted(season_batter_totals.get(season_key_a, {}).values(), reverse=True)[:3])
            top_runs_b = sum(sorted(season_batter_totals.get(season_key_b, {}).values(), reverse=True)[:3])
            top_wickets_a = sum(sorted(season_bowler_totals.get(season_key_a, {}).values(), reverse=True)[:3])
            top_wickets_b = sum(sorted(season_bowler_totals.get(season_key_b, {}).values(), reverse=True)[:3])

            computed_rows.append(
                {
                    "team_a_recent_form": round(form_a, 3),
                    "team_b_recent_form": round(form_b, 3),
                    "head_to_head_win_pct_team_a": round(h2h_pct, 3),
                    "team_a_batting_strength": round(bat_a, 2),
                    "team_b_batting_strength": round(bat_b, 2),
                    "team_a_bowling_strength": round(bowl_a, 2),
                    "team_b_bowling_strength": round(bowl_b, 2),
                    "team_a_top_run_getters_runs": round(float(top_runs_a), 2),
                    "team_b_top_run_getters_runs": round(float(top_runs_b), 2),
                    "team_a_top_wicket_takers_wickets": round(float(top_wickets_a), 2),
                    "team_b_top_wicket_takers_wickets": round(float(top_wickets_b), 2),
                    # Compute real venue advantage — cricket venues have identifiable home teams.
                    # Correctly rewards home ground familiarity that the raw cricsheet JSON omits.
                    "venue_advantage_team_a": _compute_venue_advantage(venue_name, ta, tb),
                }
            )

            # Update state *after* recording (to avoid lookahead)
            won = int(row["team_a_win"])
            recent_wins.setdefault(ta, []).append(won)
            recent_wins.setdefault(tb, []).append(1 - won)

            if ta == key[0]:
                h2h.setdefault(key, []).append(won)
            else:
                h2h.setdefault(key, []).append(1 - won)

            team_batting.setdefault(ta, []).append(float(row["_ta_runs"]))
            team_batting.setdefault(tb, []).append(float(row["_tb_runs"]))
            team_bowling.setdefault(ta, []).append(float(row["_ta_wkts"]))
            team_bowling.setdefault(tb, []).append(float(row["_tb_wkts"]))

            batter_totals_a = season_batter_totals.setdefault(season_key_a, {})
            for player, runs in row.get("_ta_batter_runs", {}).items():
                batter_totals_a[player] = batter_totals_a.get(player, 0) + int(runs)
            batter_totals_b = season_batter_totals.setdefault(season_key_b, {})
            for player, runs in row.get("_tb_batter_runs", {}).items():
                batter_totals_b[player] = batter_totals_b.get(player, 0) + int(runs)

            bowler_totals_a = season_bowler_totals.setdefault(season_key_a, {})
            for player, wickets in row.get("_ta_bowler_wickets", {}).items():
                bowler_totals_a[player] = bowler_totals_a.get(player, 0) + int(wickets)
            bowler_totals_b = season_bowler_totals.setdefault(season_key_b, {})
            for player, wickets in row.get("_tb_bowler_wickets", {}).items():
                bowler_totals_b[player] = bowler_totals_b.get(player, 0) + int(wickets)

        computed = pd.DataFrame(computed_rows, index=df.index)
        result = pd.concat([df, computed], axis=1)
        return result.drop(
            columns=[
                "match_date",
                "_ta_runs",
                "_tb_runs",
                "_ta_wkts",
                "_tb_wkts",
                "_ta_batter_runs",
                "_tb_batter_runs",
                "_ta_bowler_wickets",
                "_tb_bowler_wickets",
            ],
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


# ---------------------------------------------------------------------------
# Venue profile computation from ball-by-ball data
# ---------------------------------------------------------------------------

# Canonical name mapping so different cricsheet spellings merge correctly.
_VENUE_ALIASES: dict[str, str] = {
    "feroz shah kotla": "Arun Jaitley Stadium",
    "arun jaitley stadium": "Arun Jaitley Stadium",
    "arun jaitley stadium, delhi": "Arun Jaitley Stadium",
    "eden gardens": "Eden Gardens",
    "eden gardens, kolkata": "Eden Gardens",
    "wankhede stadium": "Wankhede Stadium",
    "wankhede stadium, mumbai": "Wankhede Stadium",
    "m chinnaswamy stadium": "M Chinnaswamy Stadium",
    "m chinnaswamy stadium, bengaluru": "M Chinnaswamy Stadium",
    "m.chinnaswamy stadium": "M Chinnaswamy Stadium",
    "ma chidambaram stadium": "MA Chidambaram Stadium",
    "ma chidambaram stadium, chepauk": "MA Chidambaram Stadium",
    "ma chidambaram stadium, chepauk, chennai": "MA Chidambaram Stadium",
    "rajiv gandhi international stadium": "Rajiv Gandhi International Stadium",
    "rajiv gandhi international stadium, uppal": "Rajiv Gandhi International Stadium",
    "rajiv gandhi international stadium, uppal, hyderabad": "Rajiv Gandhi International Stadium",
    "narendra modi stadium": "Narendra Modi Stadium",
    "narendra modi stadium, ahmedabad": "Narendra Modi Stadium",
    "sardar patel stadium, motera": "Narendra Modi Stadium",
    "sawai mansingh stadium": "Sawai Mansingh Stadium",
    "sawai mansingh stadium, jaipur": "Sawai Mansingh Stadium",
    "punjab cricket association stadium, mohali": "Punjab Cricket Association IS Bindra Stadium",
    "punjab cricket association is bindra stadium": "Punjab Cricket Association IS Bindra Stadium",
    "punjab cricket association is bindra stadium, mohali": "Punjab Cricket Association IS Bindra Stadium",
    "punjab cricket association is bindra stadium, mohali, chandigarh": "Punjab Cricket Association IS Bindra Stadium",
    "bharat ratna shri atal bihari vajpayee ekana cricket stadium, lucknow": "BRSABV Ekana Cricket Stadium",
    "brsabv ekana cricket stadium": "BRSABV Ekana Cricket Stadium",
    "himachal pradesh cricket association stadium": "Himachal Pradesh Cricket Association Stadium",
    "himachal pradesh cricket association stadium, dharamsala": "Himachal Pradesh Cricket Association Stadium",
    "dr dy patil sports academy": "DY Patil Stadium",
    "dr dy patil sports academy, mumbai": "DY Patil Stadium",
    "dy patil stadium": "DY Patil Stadium",
    "brabourne stadium": "Brabourne Stadium",
    "brabourne stadium, mumbai": "Brabourne Stadium",
    "maharashtra cricket association stadium": "Maharashtra Cricket Association Stadium",
    "maharashtra cricket association stadium, pune": "Maharashtra Cricket Association Stadium",
    "subrata roy sahara stadium": "Maharashtra Cricket Association Stadium",
    "dr. y.s. rajasekhara reddy aca-vdca cricket stadium": "ACA-VDCA Cricket Stadium",
    "dr. y.s. rajasekhara reddy aca-vdca cricket stadium, visakhapatnam": "ACA-VDCA Cricket Stadium",
    "aca-vdca cricket stadium": "ACA-VDCA Cricket Stadium",
    "saurashtra cricket association stadium": "Saurashtra Cricket Association Stadium",
    "holkar cricket stadium": "Holkar Cricket Stadium",
    "maharaja yadavindra singh international cricket stadium, mullanpur": "Maharaja Yadavindra Singh International Cricket Stadium",
    "maharaja yadavindra singh international cricket stadium, new chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium",
    "barsapara cricket stadium, guwahati": "Barsapara Cricket Stadium",
    "jsca international stadium complex": "JSCA International Stadium Complex",
    "raipur international cricket stadium": "Raipur International Cricket Stadium",
    "shaheed veer narayan singh international stadium": "Raipur International Cricket Stadium",
}

# Bowler name patterns that signal spin vs pace (cricsheet uses initials + surname)
_KNOWN_SPIN_BOWLERS: set[str] = set()  # populated at runtime if needed
_SPIN_BOWLER_KIND_KEYWORDS = {"lbg", "sla", "ob", "lb", "chinaman"}

# Well-known T20/IPL spin bowlers — surnames (lowercased) for fuzzy matching.
# Covers the most common spinners across IPL history.  The classifier falls
# back to "pace" for unknown names so only false-negatives are possible.
_SPIN_BOWLER_SURNAMES: set[str] = {
    # Leg-spin / wrist-spin
    "chahal", "rashid", "zampa", "tahir", "lamichhane", "mishra", "kumble",
    "tanveer", "qadir", "devdutt", "ravi bishnoi", "bishnoi", "rahul chahar",
    "wanindu", "hasaranga", "theekshana", "sodhi", "shamsi", "ish sodhi",
    "sandeep lamichhane", "amit mishra",
    # Off-spin / finger-spin
    "ashwin", "narine", "sundar", "jadeja", "axar", "moeen", "santner",
    "harbhajan", "swann", "lyon", "mujeeb", "hogg", "murali", "muralitharan",
    "shakib", "krunal", "gowtham", "rishi dhawan", "kuldeep",
    "washington sundar", "ravichandran ashwin", "sunil narine",
    "ravindra jadeja", "axar patel", "moeen ali", "mitchell santner",
    "krunal pandya", "krishnappa gowtham", "kuldeep yadav",
    "harbhajan singh", "brad hogg", "muttiah muralitharan",
    "shakib al hasan", "mujeeb ur rahman",
    # Chinaman / SLA
    "kuldeep", "chinaman",
    # Recent IPL spinners
    "varun chakravarthy", "chakravarthy", "ravi ashwin", "noor ahmad",
    "maheesh theekshana", "rachin ravindra", "riyan parag",
    "mahesh theekshana", "piyush chawla", "chawla", "pravin tambe", "tambe",
    "imran tahir", "yuzvendra chahal", "r ashwin", "pp chawla",
    "sk raina", "sp narine", "sl malinga",
    # Cricsheet initials + surname format
    "yuzvendra", "rashid khan", "varun", "r bishnoi",
}

# Full-name lookup built lazily to avoid O(N) search per delivery.
_SPIN_NAME_CACHE: dict[str, bool] = {}


def _is_known_spinner(bowler: str) -> bool:
    """Return True if *bowler* (cricsheet format) matches a known spinner."""
    if bowler in _SPIN_NAME_CACHE:
        return _SPIN_NAME_CACHE[bowler]

    lower = bowler.lower().strip()
    # Direct surname match
    parts = lower.split()
    result = False
    if any(part in _SPIN_BOWLER_SURNAMES for part in parts):
        result = True
    elif lower in _SPIN_BOWLER_SURNAMES:
        result = True
    else:
        # Try without initials: "YS Chahal" → "chahal"
        surname = parts[-1] if parts else ""
        if surname in _SPIN_BOWLER_SURNAMES:
            result = True

    _SPIN_NAME_CACHE[bowler] = result
    return result


def _normalise_venue(raw: str) -> str:
    key = raw.strip().lower()
    return _VENUE_ALIASES.get(key, raw.strip())


def compute_venue_profiles(json_dirs: list[Path], min_matches: int = 3) -> dict[str, dict[str, float]]:
    """Compute venue behavioral profiles from cricsheet JSON ball-by-ball data.

    Returns a dict keyed by canonical venue name with values matching the
    ``VenueProfile`` schema used by the feature pipeline.
    """
    from collections import defaultdict

    # Accumulators per venue
    first_innings_totals: dict[str, list[float]] = defaultdict(list)
    chase_wins: dict[str, list[int]] = defaultdict(list)  # 1 = chaser won, 0 = setter won
    deliveries_count: dict[str, int] = defaultdict(int)
    boundaries_count: dict[str, int] = defaultdict(int)
    spin_wickets: dict[str, int] = defaultdict(int)
    pace_wickets: dict[str, int] = defaultdict(int)
    spin_runs: dict[str, float] = defaultdict(float)
    spin_balls: dict[str, int] = defaultdict(int)
    pace_runs: dict[str, float] = defaultdict(float)
    pace_balls: dict[str, int] = defaultdict(int)

    for directory in json_dirs:
        for path in sorted(directory.rglob("*.json")):
            if path.name == _META_FILENAME:
                continue
            try:
                _process_match_for_venue(
                    path,
                    first_innings_totals, chase_wins,
                    deliveries_count, boundaries_count,
                    spin_wickets, pace_wickets,
                    spin_runs, spin_balls,
                    pace_runs, pace_balls,
                )
            except Exception:
                continue

    profiles: dict[str, dict[str, float]] = {}
    for venue in first_innings_totals:
        match_count = len(first_innings_totals[venue])
        if match_count < min_matches:
            continue
        total_wickets = spin_wickets[venue] + pace_wickets[venue]
        profiles[venue] = {
            "avg_first_innings_score": round(
                sum(first_innings_totals[venue]) / len(first_innings_totals[venue]), 2
            ),
            "chase_win_pct": round(
                sum(chase_wins[venue]) / len(chase_wins[venue]), 4
            ) if chase_wins[venue] else 0.5,
            "spin_wicket_pct": round(
                spin_wickets[venue] / total_wickets, 4
            ) if total_wickets > 0 else 0.4,
            "pace_wicket_pct": round(
                pace_wickets[venue] / total_wickets, 4
            ) if total_wickets > 0 else 0.6,
            "boundary_rate": round(
                boundaries_count[venue] / max(deliveries_count[venue], 1), 4
            ),
            "spin_economy": round(
                spin_runs[venue] / max(spin_balls[venue], 1) * 6, 4
            ) if spin_balls[venue] else 8.2,
            "pace_economy": round(
                pace_runs[venue] / max(pace_balls[venue], 1) * 6, 4
            ) if pace_balls[venue] else 9.0,
            "matches": match_count,
        }
    return dict(sorted(profiles.items()))


def _process_match_for_venue(
    path: Path,
    first_innings_totals: dict[str, list[float]],
    chase_wins: dict[str, list[int]],
    deliveries_count: dict[str, int],
    boundaries_count: dict[str, int],
    spin_wickets: dict[str, int],
    pace_wickets: dict[str, int],
    spin_runs: dict[str, float],
    spin_balls: dict[str, int],
    pace_runs: dict[str, float],
    pace_balls: dict[str, int],
) -> None:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    info = data.get("info", {})
    venue_raw = info.get("venue") or info.get("city", "")
    if not venue_raw:
        return
    venue = _normalise_venue(venue_raw)

    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    if not winner or outcome.get("result") in ("no result", "tie"):
        return

    innings_list = data.get("innings", [])
    if len(innings_list) < 2:
        return

    # Build registry of bowler types from the match info if available
    bowler_type_registry: dict[str, str] = {}  # bowler name → "spin" | "pace"
    registry = info.get("registry", {}).get("people", {})
    players_info = info.get("players", {})

    # Track innings totals
    innings_totals: list[tuple[str, int]] = []
    for innings_idx, innings in enumerate(innings_list):
        batting_team = innings.get("team", "")
        total = 0
        for over in innings.get("overs", []):
            for delivery in over.get("deliveries", []):
                runs = delivery.get("runs", {})
                total_runs = runs.get("total", 0)
                batter_runs = runs.get("batter", 0)
                total += total_runs
                deliveries_count[venue] += 1

                # Boundary detection
                if batter_runs in (4, 6):
                    boundaries_count[venue] += 1

                # Bowler classification
                bowler = delivery.get("bowler", "")
                bowler_category = _classify_bowler(bowler, bowler_type_registry, info)

                # Economy tracking
                if bowler_category == "spin":
                    spin_runs[venue] += total_runs
                    spin_balls[venue] += 1
                else:
                    pace_runs[venue] += total_runs
                    pace_balls[venue] += 1

                # Wicket tracking
                for wicket in delivery.get("wickets", []):
                    kind = str(wicket.get("kind", "")).lower()
                    if kind in ("run out", "retired hurt", "retired out", "obstructing the field"):
                        continue
                    if bowler_category == "spin":
                        spin_wickets[venue] += 1
                    else:
                        pace_wickets[venue] += 1

        innings_totals.append((batting_team, total))

    # First innings score
    if innings_totals:
        first_innings_totals[venue].append(float(innings_totals[0][1]))

    # Chase win: did the team batting second win?
    if len(innings_totals) >= 2:
        second_batting_team = innings_totals[1][0]
        chase_wins[venue].append(int(winner == second_batting_team))


def _classify_bowler(bowler: str, registry: dict[str, str], info: dict) -> str:
    """Classify a bowler as 'spin' or 'pace' using a known-spinners list."""
    if bowler in registry:
        return registry[bowler]

    result = "spin" if _is_known_spinner(bowler) else "pace"
    registry[bowler] = result
    return result
