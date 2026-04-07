"""Normalize a ball-by-ball CSV into the standard IPL CSV files.

Converts alternate Kaggle ball-by-ball datasets (e.g. sujalninawe/
ipl-2026-ball-by-ball-dataset-daily-updated) into the canonical file
layout that ``IplCsvDataProvider`` expects:

  matches.csv, deliveries.csv, points_table.csv,
  orange_cap.csv, purple_cap.csv, squads.csv

Usage
-----
  python scripts/normalize_bbb_csv.py <input.csv> <output_dir>

The script auto-detects the column layout via flexible aliases so it can
handle multiple ball-by-ball providers without code changes.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Column alias resolution
# ---------------------------------------------------------------------------

_ALIASES: dict[str, tuple[str, ...]] = {
    "match_id": ("match_id", "matchid", "id"),
    "match_name": ("match_name", "matchname", "fixture", "match"),
    "date": ("date", "match_date", "start_date"),
    "innings": ("innings", "inning", "innings_no"),
    "batting_team": ("batting_team", "battingteam", "bat_team"),
    "bowling_team": ("bowling_team", "bowlingteam", "bowl_team"),
    "batter": ("batter", "batsman", "striker"),
    "bowler": ("bowler",),
    "run": ("run", "runs", "batsman_runs", "batter_runs"),
    "ball": ("ball", "ball_no", "ball_number"),
    "extras": ("extras", "extra_runs", "extra"),
    "boundary": ("boundary", "is_boundary"),
    "over": ("over", "overs"),
    "wicket": ("wicket", "is_wicket", "wickets"),
    "total_runs": ("total_runs", "totalrun", "cumulative_runs", "total"),
}


def _resolve_columns(header: list[str]) -> dict[str, str]:
    """Map canonical names -> actual CSV column names."""
    normalised = {re.sub(r"[^a-z0-9]", "_", h.strip().lower()): h for h in header}
    mapping: dict[str, str] = {}
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            key = re.sub(r"[^a-z0-9]", "_", alias)
            if key in normalised:
                mapping[canonical] = normalised[key]
                break
    return mapping


# ---------------------------------------------------------------------------
# Match result detection
# ---------------------------------------------------------------------------

def _parse_match_name(name: str) -> tuple[str, str]:
    """Extract two team short-codes from 'RCB VS SRH' style names."""
    parts = re.split(r"\s+(?:vs|v)\s+", name.strip(), flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return name.strip(), ""


_SHORT_TO_FULL: dict[str, str] = {
    "CSK": "Chennai Super Kings",
    "DC": "Delhi Capitals",
    "GT": "Gujarat Titans",
    "KKR": "Kolkata Knight Riders",
    "LSG": "Lucknow Super Giants",
    "MI": "Mumbai Indians",
    "PBKS": "Punjab Kings",
    "RCB": "Royal Challengers Bengaluru",
    "RR": "Rajasthan Royals",
    "SRH": "Sunrisers Hyderabad",
}


def _expand_team(short: str) -> str:
    return _SHORT_TO_FULL.get(short.upper().strip(), short.strip())


def _normalise_date(raw: str) -> str:
    """Convert '28-Mar-26' or similar formats to 'YYYY-MM-DD'."""
    import datetime

    for fmt in ("%d-%b-%y", "%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw.strip()


# ---------------------------------------------------------------------------
# Main normalisation
# ---------------------------------------------------------------------------

def normalise(input_path: Path, output_dir: Path) -> dict[str, int]:
    """Read a ball-by-ball CSV and write the standard IPL CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col_map = _resolve_columns(header)
        rows = list(reader)

    if not rows:
        raise SystemExit("Input CSV is empty")

    required = {"match_id", "innings", "batting_team", "bowling_team", "batter", "bowler"}
    missing = required - set(col_map)
    if missing:
        raise SystemExit(f"Cannot resolve required columns: {', '.join(sorted(missing))}")

    def _get(row: list[str], key: str, default: str = "") -> str:
        col_name = col_map.get(key)
        if col_name is None:
            return default
        idx = header.index(col_name)
        return row[idx].strip() if idx < len(row) else default

    # -----------------------------------------------------------------------
    # Pass 1: group by match
    # -----------------------------------------------------------------------

    matches: dict[str, dict] = {}  # match_id -> match info
    deliveries: list[dict] = []
    batter_runs: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # team -> player -> runs
    bowler_wickets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # team -> player -> wickets
    team_players: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        mid = _get(row, "match_id")
        innings_raw = _get(row, "innings")
        innings_num = re.sub(r"[^0-9]", "", innings_raw) or "1"
        bat_team = _expand_team(_get(row, "batting_team"))
        bowl_team = _expand_team(_get(row, "bowling_team"))
        batter_name = _get(row, "batter")
        bowler_name = _get(row, "bowler")
        run_val = int(_get(row, "run", "0") or "0")
        extras_raw = _get(row, "extras", "0")
        # extras may be a type code (WD, NB, BYE, LB, B4) or a number
        try:
            extras_val = int(extras_raw)
        except ValueError:
            # Type codes mean 1 extra run was added (wide/no-ball/bye etc.)
            extras_val = 1 if extras_raw else 0
        wicket_raw = _get(row, "wicket", "0")
        wicket_cumulative = int(wicket_raw) if wicket_raw.isdigit() else 0
        total_val = _get(row, "total_runs", "")
        match_name = _get(row, "match_name", "")
        match_date = _get(row, "date", "")

        # Build match info
        if mid not in matches:
            team_a_short, team_b_short = _parse_match_name(match_name)
            matches[mid] = {
                "match_id": mid,
                "date": _normalise_date(match_date),
                "team_a": _expand_team(team_a_short) if team_a_short else bat_team,
                "team_b": _expand_team(team_b_short) if team_b_short else bowl_team,
                "innings": {},
            }

        minfo = matches[mid]
        inn_key = innings_num
        if inn_key not in minfo["innings"]:
            minfo["innings"][inn_key] = {"batting_team": bat_team, "total": 0, "wickets": 0, "prev_wickets": 0}
        inn = minfo["innings"][inn_key]

        # Update innings totals from running total or per-ball
        if total_val:
            current_total = int(total_val)
            if current_total > inn["total"]:
                inn["total"] = current_total
        else:
            inn["total"] += run_val + extras_val

        # Wicket column may be cumulative (0,0,1,1,1,2,...) — detect per-ball wicket
        ball_wicket = 0
        if wicket_cumulative > inn["prev_wickets"]:
            ball_wicket = wicket_cumulative - inn["prev_wickets"]
            inn["wickets"] = wicket_cumulative
            inn["prev_wickets"] = wicket_cumulative

        # Deliveries row
        deliveries.append({
            "match_id": mid,
            "innings": innings_num,
            "batting_team": bat_team,
            "bowling_team": bowl_team,
            "batter": batter_name,
            "bowler": bowler_name,
            "batsman_runs": run_val,
            "extra_runs": extras_val,
            "total_runs": run_val + extras_val,
            "is_wicket": 1 if ball_wicket > 0 else 0,
        })

        # Player stats
        batter_runs[bat_team][batter_name] += run_val
        if ball_wicket:
            bowler_wickets[bowl_team][bowler_name] += ball_wicket

        team_players[bat_team].add(batter_name)
        team_players[bowl_team].add(bowler_name)

    # -----------------------------------------------------------------------
    # Determine match winners
    # -----------------------------------------------------------------------

    match_results: list[dict] = []
    team_wins: dict[str, int] = defaultdict(int)
    team_losses: dict[str, int] = defaultdict(int)
    team_form: dict[str, list[str]] = defaultdict(list)

    for mid in sorted(matches, key=lambda k: int(k)):
        m = matches[mid]
        innings = m["innings"]
        if len(innings) < 2:
            continue

        first = innings.get("1", {})
        second = innings.get("2", {})

        first_total = first.get("total", 0)
        second_total = second.get("total", 0)
        first_bat = first.get("batting_team", m["team_a"])
        second_bat = second.get("batting_team", m["team_b"])

        if second_total > first_total:
            winner = second_bat
        elif first_total > second_total:
            winner = first_bat
        else:
            winner = ""  # tie

        match_results.append({
            "match_id": mid,
            "date": m["date"],
            "team_a": m["team_a"],
            "team_b": m["team_b"],
            "winner": winner,
            "status": "completed" if winner else "tie",
        })

        if winner:
            loser = m["team_b"] if winner == m["team_a"] else m["team_a"]
            team_wins[winner] += 1
            team_losses[loser] += 1
            team_form[winner].append("W")
            team_form[loser].append("L")

    # -----------------------------------------------------------------------
    # Write matches.csv
    # -----------------------------------------------------------------------

    matches_path = output_dir / "matches.csv"
    with matches_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["match_id", "date", "team_a", "team_b", "winner", "status"])
        w.writeheader()
        w.writerows(match_results)

    # -----------------------------------------------------------------------
    # Write deliveries.csv
    # -----------------------------------------------------------------------

    deliv_path = output_dir / "deliveries.csv"
    deliv_fields = ["match_id", "innings", "batting_team", "bowling_team", "batter", "bowler",
                    "batsman_runs", "extra_runs", "total_runs", "is_wicket"]
    with deliv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=deliv_fields)
        w.writeheader()
        w.writerows(deliveries)

    # -----------------------------------------------------------------------
    # Write points_table.csv
    # -----------------------------------------------------------------------

    all_teams = sorted(set(team_wins.keys()) | set(team_losses.keys()))
    pts_path = output_dir / "points_table.csv"
    with pts_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["team", "played", "won", "form"])
        w.writeheader()
        for team in all_teams:
            played = team_wins[team] + team_losses[team]
            form_str = " ".join(team_form.get(team, [])[-5:])
            w.writerow({"team": team, "played": played, "won": team_wins[team], "form": form_str})

    # -----------------------------------------------------------------------
    # Write orange_cap.csv (top run scorers)
    # -----------------------------------------------------------------------

    oc_path = output_dir / "orange_cap.csv"
    with oc_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["team", "player", "runs"])
        w.writeheader()
        for team in sorted(batter_runs):
            for player, runs in sorted(batter_runs[team].items(), key=lambda x: -x[1]):
                w.writerow({"team": team, "player": player, "runs": runs})

    # -----------------------------------------------------------------------
    # Write purple_cap.csv (top wicket takers)
    # -----------------------------------------------------------------------

    pc_path = output_dir / "purple_cap.csv"
    with pc_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["team", "player", "wickets"])
        w.writeheader()
        for team in sorted(bowler_wickets):
            for player, wickets in sorted(bowler_wickets[team].items(), key=lambda x: -x[1]):
                w.writerow({"team": team, "player": player, "wickets": wickets})

    # -----------------------------------------------------------------------
    # Write squads.csv
    # -----------------------------------------------------------------------

    sq_path = output_dir / "squads.csv"
    with sq_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["team", "player"])
        w.writeheader()
        for team in sorted(team_players):
            for player in sorted(team_players[team]):
                w.writerow({"team": team, "player": player})

    return {
        "matches": len(match_results),
        "deliveries": len(deliveries),
        "teams": len(all_teams),
        "batters": sum(len(v) for v in batter_runs.values()),
        "bowlers": sum(len(v) for v in bowler_wickets.values()),
    }


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.csv> <output_dir>")
        raise SystemExit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    stats = normalise(input_path, output_dir)
    print(f"Normalised {input_path.name} → {output_dir}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
