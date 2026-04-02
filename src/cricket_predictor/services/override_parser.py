"""Match context override parser.

Parses free-text coaching notes like:
  "Pat Cummins is injured"
  "SRH missing Cummins and Klaasen"
  "Eden Gardens is a batting pitch today"
  "KKR bowling attack is weakened"

Returns a list of structured adjustment dicts that prediction_tracker applies
as multipliers on top of the base bowling/batting strength features.

Supported patterns
------------------
INJURY / ABSENCE
  "<player> is injured|injured|out|missing|unavailable|withdrawn|doubtful"
  "<team> missing <player>(, <player>)*"
  "<team> without <player>"

PITCH OVERRIDE
  "<venue> is a (batting|bowling|spin|pace) pitch"
  "pitch at <venue> favours (batters|bowlers|spinners|pacers)"

TEAM STRENGTH
  "<team> bowling|batting (is|looks) (weak|strong|poor|dominant)"
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data helpers — lazy-loaded from squad_profiles.json
# ---------------------------------------------------------------------------

_squad_cache: dict | None = None


def _get_squad() -> dict:
    global _squad_cache
    if _squad_cache is None:
        from pathlib import Path
        cache = Path(__file__).parents[3] / "data" / "squad_profiles.json"
        if cache.exists():
            _squad_cache = json.loads(cache.read_text())
        else:
            _squad_cache = {}
    return _squad_cache


def _find_player(name: str) -> dict | None:
    """Return {team, name, role} for a player name (fuzzy, case-insensitive)."""
    needle = name.strip().lower()
    squad = _get_squad()
    for team, data in squad.items():
        for p in data.get("players", []):
            pname = p["name"].lower()
            # Match on full name or last name
            if needle in pname or pname.endswith(needle):
                return {"team": team, "name": p["name"], "role": p["role"]}
    return None


# ---------------------------------------------------------------------------
# Adjustment factors
# ---------------------------------------------------------------------------

# How much to reduce a strength score when a key player is absent
_ROLE_BOWLING_IMPACT = {
    "Bowler":      0.12,   # specialist bowler missing → -12% bowling strength
    "All-Rounder": 0.06,
    "Batter":      0.00,
    "WK-Batter":   0.00,
}
_ROLE_BATTING_IMPACT = {
    "Batter":      0.10,
    "WK-Batter":   0.08,
    "All-Rounder": 0.05,
    "Bowler":      0.01,
}

_STRENGTH_SENTIMENT = {
    # positive words → +multiplier on the stated dimension
    "strong": +0.10, "good": +0.08, "dominant": +0.15, "excellent": +0.12,
    # negative words → -multiplier
    "weak": -0.12, "poor": -0.10, "bad": -0.12, "terrible": -0.15,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_override(text: str) -> list[dict]:
    """Parse free-text note into a list of structured adjustment dicts.

    Each dict has:
        type        : "injury" | "pitch" | "team_strength"
        team        : canonical team name (when applicable)
        player      : player name (injury only)
        role        : player role (injury only)
        bowl_delta  : float multiplier delta applied to bowling_strength (0–1 scale)
        bat_delta   : float multiplier delta applied to batting_strength (0–1 scale)
        description : human-readable summary
    """
    adjustments: list[dict] = []
    lines = [l.strip() for l in re.split(r"[.\n;,]", text) if l.strip()]

    for line in lines:
        lower = line.lower()

        # ── Injury / absence patterns ──────────────────────────────────
        injury_keywords = r"injured|injury|retired hurt|out|unavailable|withdrawn|doubtful|not playing|absent|missing|ruled out|won't play|will not play"
        injury_match = re.search(
            rf"([A-Z][a-zA-Z ]+?)(?:\s+is|\s+has|\s+was)?\s+(?:{injury_keywords})",
            line, re.IGNORECASE,
        )
        # "missing <player>" pattern
        missing_match = re.search(
            r"missing\s+([A-Z][a-zA-Z ]+?)(?:\s+and\s+([A-Z][a-zA-Z ]+?))?(?:\s|$)",
            line, re.IGNORECASE,
        )
        # "without <player>"
        without_match = re.search(r"without\s+([A-Z][a-zA-Z ]+?)(?:\s|$)", line, re.IGNORECASE)

        candidates: list[str] = []
        if injury_match:
            candidates.append(injury_match.group(1).strip())
        if missing_match:
            candidates.append(missing_match.group(1).strip())
            if missing_match.group(2):
                candidates.append(missing_match.group(2).strip())
        if without_match:
            candidates.append(without_match.group(1).strip())

        for cand in candidates:
            player = _find_player(cand)
            if player:
                role = player["role"]
                adj = {
                    "type": "injury",
                    "team": player["team"],
                    "player": player["name"],
                    "role": role,
                    "bowl_delta": -_ROLE_BOWLING_IMPACT.get(role, 0.0),
                    "bat_delta":  -_ROLE_BATTING_IMPACT.get(role, 0.0),
                    "description": f"🚑 {player['name']} ({role}, {player['team']}) unavailable",
                }
                adjustments.append(adj)
                log.info("Override parsed: %s", adj["description"])
            else:
                log.debug("Player not found in squad: %r", cand)

        # ── Pitch / conditions override ────────────────────────────────
        pitch_match = re.search(
            r"(?:pitch|track|surface|wicket)\b.{0,50}?\b(batting|bowling|spin|pace|flat|seaming|turning)",
            lower,
        ) or re.search(
            r"(batting|bowling|spin|pace|flat|seaming|turning)\b.{0,30}?\bpitch\b",
            lower,
        )
        if pitch_match:
            nature = pitch_match.group(1)
            pitch_map = {
                "batting": ("pitch_batting", +0.5),
                "flat":    ("pitch_batting", +0.4),
                "bowling": ("pitch_batting", -0.5),
                "seaming": ("pitch_batting", -0.4),
                "spin":    ("spin_effectiveness", +0.3),
                "turning": ("spin_effectiveness", +0.35),
                "pace":    ("pace_effectiveness", +0.3),
            }
            if nature in pitch_map:
                key, val = pitch_map[nature]
                adjustments.append({
                    "type": "pitch",
                    "team": None,
                    "player": None,
                    "role": None,
                    "bowl_delta": 0.0,
                    "bat_delta": 0.0,
                    "pitch_key": key,
                    "pitch_val": val,
                    "description": f"🏟️ Pitch note: {nature} — {key} override {val:+.1f}",
                })

        # ── Team strength sentiment ────────────────────────────────────
        for team_name in (
            "csk", "kkr", "mi", "srh", "rcb", "dc", "gt", "rr", "lsg", "pbks",
            "chennai", "kolkata", "mumbai", "hyderabad", "bangalore", "bengaluru",
            "delhi", "gujarat", "rajasthan", "lucknow", "punjab",
        ):
            if team_name not in lower:
                continue

            for dim in ("bowling", "batting"):
                if dim not in lower:
                    continue
                for sentiment, factor in _STRENGTH_SENTIMENT.items():
                    if sentiment in lower:
                        from cricket_predictor.providers.cricinfo_standings import resolve_team_name
                        canonical = resolve_team_name(team_name)
                        adjustments.append({
                            "type": "team_strength",
                            "team": canonical,
                            "player": None,
                            "role": None,
                            "bowl_delta": factor if dim == "bowling" else 0.0,
                            "bat_delta":  factor if dim == "batting"  else 0.0,
                            "description": f"📊 {canonical} {dim} noted as {sentiment} ({factor:+.0%})",
                        })
                        break

    return adjustments


def apply_overrides(
    team_a: str,
    team_b: str,
    team_a_bat: float,
    team_b_bat: float,
    team_a_bowl: float,
    team_b_bowl: float,
    overrides: list[dict],
) -> tuple[float, float, float, float]:
    """Apply parsed override adjustments to base strength values.

    Returns (team_a_bat, team_b_bat, team_a_bowl, team_b_bowl) after adjustments.
    Strengths are on a 40–100 scale; deltas are fractional (e.g. -0.12 = -12%).
    """
    # Accumulate deltas per team
    deltas: dict[str, dict[str, float]] = {
        team_a: {"bat": 0.0, "bowl": 0.0},
        team_b: {"bat": 0.0, "bowl": 0.0},
    }

    for ov in overrides:
        team = ov.get("team")
        if team not in deltas:
            continue
        deltas[team]["bat"]  += ov.get("bat_delta",  0.0)
        deltas[team]["bowl"] += ov.get("bowl_delta", 0.0)

    def _apply(base: float, delta: float) -> float:
        # Delta is a fraction of the current value; clamp result to [25, 100]
        return round(max(25.0, min(100.0, base * (1.0 + delta))), 2)

    return (
        _apply(team_a_bat,  deltas[team_a]["bat"]),
        _apply(team_b_bat,  deltas[team_b]["bat"]),
        _apply(team_a_bowl, deltas[team_a]["bowl"]),
        _apply(team_b_bowl, deltas[team_b]["bowl"]),
    )
