"""Azure OpenAI LLM provider — generates pre-match analysis via GPT-4.1.

Uses httpx (already a project dependency) to call the Azure OpenAI
chat/completions endpoint.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
import ssl

import httpx

from cricket_predictor.config.settings import get_settings

log = logging.getLogger(__name__)

_TIMEOUT = 30  # seconds


def _get_ssl_context():
    """Use truststore for system CA certs when available (macOS Keychain, etc.)."""
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        pass
    return True  # httpx default verification


def _build_prompt(match_context: dict) -> str:
    """Build a cricket-analyst prompt from structured match context."""
    team_a = match_context.get("team_a", "Team A")
    team_b = match_context.get("team_b", "Team B")
    venue = match_context.get("venue", "Unknown")
    match_date = match_context.get("match_date", "")
    ta_bat = match_context.get("team_a_batting", 65)
    tb_bat = match_context.get("team_b_batting", 65)
    ta_bowl = match_context.get("team_a_bowling", 65)
    tb_bowl = match_context.get("team_b_bowling", 65)
    ta_form = match_context.get("team_a_form", 50)
    tb_form = match_context.get("team_b_form", 50)
    venue_adv = match_context.get("venue_advantage", 0)
    predicted_winner = match_context.get("predicted_winner", "")
    win_pct = match_context.get("win_probability", 50)
    injuries = match_context.get("injuries", "None reported")
    overrides = match_context.get("overrides", "None")
    verified_team_a_squad = _format_player_names(match_context.get("verified_team_a_squad", []))
    verified_team_b_squad = _format_player_names(match_context.get("verified_team_b_squad", []))
    team_a_batting_leaders = _format_player_names(match_context.get("verified_team_a_batting_leaders", []))
    team_b_batting_leaders = _format_player_names(match_context.get("verified_team_b_batting_leaders", []))
    team_a_bowling_leaders = _format_player_names(match_context.get("verified_team_a_bowling_leaders", []))
    team_b_bowling_leaders = _format_player_names(match_context.get("verified_team_b_bowling_leaders", []))

    home_note = ""
    if venue_adv > 0:
        home_note = f"{team_a} has home advantage at this venue."
    elif venue_adv < 0:
        home_note = f"{team_b} has home advantage at this venue."

    return f"""**Match:** {team_a} vs {team_b}
**Date:** {match_date}
**Venue:** {venue}
{f'**Home Advantage:** {home_note}' if home_note else ''}

**Team Strengths (0-100 scale):**
- {team_a}: Batting {ta_bat:.1f}, Bowling {ta_bowl:.1f}, Recent Form {ta_form:.1f}
- {team_b}: Batting {tb_bat:.1f}, Bowling {tb_bowl:.1f}, Recent Form {tb_form:.1f}

**Injuries/Unavailability:** {injuries}
**Additional Context:** {overrides}

**Verified Current Squad And Leader Data:**
- {team_a} current squad: {verified_team_a_squad}
- {team_b} current squad: {verified_team_b_squad}
- {team_a} top current-season batters: {team_a_batting_leaders}
- {team_b} top current-season batters: {team_b_batting_leaders}
- {team_a} top current-season wicket-takers: {team_a_bowling_leaders}
- {team_b} top current-season wicket-takers: {team_b_bowling_leaders}

**ML Model Prediction:** {predicted_winner} ({win_pct:.1f}% probability)

Write a short pre-match analysis (4-5 bullet points) covering:
1. Key matchup factors (batting vs bowling strengths)
2. Venue/conditions impact
3. Player availability impact (injuries)
4. Recent form momentum
5. Your verdict — agree or nuance the ML prediction

Rules:
- Mention player names only if they appear in the verified squad or verified leader lists above.
- If the verified lists are empty for a team, keep the analysis team-level and do not guess player names.
- Do not mention outdated transfers, former IPL players, or unverified squad associations.

Keep each bullet to 1-2 sentences. Use cricket terminology. Be specific to these teams. Do NOT use markdown headers. Use plain bullet points with • character."""


_SYSTEM_PROMPT = (
    "You are an expert IPL cricket analyst. Provide concise, insightful "
    "pre-match analysis for IPL 2026 matches. Be specific, use cricket terminology, "
    "and never mention a player unless that player appears in the verified squad or leader lists provided."
)


def _format_player_names(value: object) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return "No verified player list available"
    players = [str(player).strip() for player in value if str(player).strip()]
    if not players:
        return "No verified player list available"
    return ", ".join(players)


def generate_match_analysis(match_context: dict) -> str | None:
    """Call Azure OpenAI to generate pre-match analysis.

    Returns the analysis text, or None if the API call fails or no key is configured.
    """
    settings = get_settings()
    api_key = settings.azure_openai_api_key
    endpoint = settings.azure_openai_endpoint
    api_version = settings.azure_openai_api_version
    deployment = settings.azure_openai_deployment

    if not api_key or not endpoint:
        log.warning("Azure OpenAI not configured — skipping analysis.")
        return None

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions"
    log.info("Azure OpenAI: calling %s for %s vs %s", deployment,
             match_context.get("team_a"), match_context.get("team_b"))

    prompt = _build_prompt(match_context)

    payload = {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 512,
    }

    try:
        resp = httpx.post(
            url,
            params={"api-version": api_version},
            headers={
                "Content-Type": "application/json",
                "api-key": api_key,
            },
            json=payload,
            timeout=_TIMEOUT,
            verify=_get_ssl_context(),
        )
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            log.warning("Azure OpenAI returned no choices.")
            return None

        text = choices[0].get("message", {}).get("content", "").strip()
        if not text:
            log.warning("Azure OpenAI returned empty content.")
            return None

        log.info("Azure OpenAI: got %d chars analysis for %s vs %s",
                 len(text), match_context.get("team_a"), match_context.get("team_b"))
        return text

    except httpx.HTTPStatusError as exc:
        log.warning("Azure OpenAI API error %s: %s", exc.response.status_code, exc.response.text[:200])
        return None
    except Exception as exc:
        log.warning("Azure OpenAI call failed: %s", exc)
        return None
