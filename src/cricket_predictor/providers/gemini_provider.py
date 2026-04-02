"""Gemini LLM provider — generates pre-match analysis via the REST API.

Uses httpx (already a project dependency) to call the Gemini generateContent
endpoint directly, avoiding the heavy google-generativeai SDK and its
dependency issues on Python 3.14.
"""

from __future__ import annotations

import logging
import ssl
from functools import lru_cache

import httpx

from cricket_predictor.config.settings import get_settings

log = logging.getLogger(__name__)

_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_TIMEOUT = 30  # seconds


def _get_ssl_context() -> ssl.SSLContext:
    """Use truststore for system CA certs when available (macOS Keychain, etc.)."""
    try:
        import truststore
        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return ctx
    except Exception:
        pass
    # Fallback: standard Python SSL (works on Render / Linux)
    return True  # type: ignore[return-value]  # httpx accepts True for default verification


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

    home_note = ""
    if venue_adv > 0:
        home_note = f"{team_a} has home advantage at this venue."
    elif venue_adv < 0:
        home_note = f"{team_b} has home advantage at this venue."

    return f"""You are an expert IPL cricket analyst. Provide a concise pre-match analysis for the upcoming IPL 2026 match.

**Match:** {team_a} vs {team_b}
**Date:** {match_date}
**Venue:** {venue}
{f'**Home Advantage:** {home_note}' if home_note else ''}

**Team Strengths (0-100 scale):**
- {team_a}: Batting {ta_bat:.1f}, Bowling {ta_bowl:.1f}, Recent Form {ta_form:.1f}
- {team_b}: Batting {tb_bat:.1f}, Bowling {tb_bowl:.1f}, Recent Form {tb_form:.1f}

**Injuries/Unavailability:** {injuries}
**Additional Context:** {overrides}

**ML Model Prediction:** {predicted_winner} ({win_pct:.1f}% probability)

Write a short pre-match analysis (4-5 bullet points) covering:
1. Key matchup factors (batting vs bowling strengths)
2. Venue/conditions impact
3. Player availability impact (injuries)
4. Recent form momentum
5. Your verdict — agree or nuance the ML prediction

Keep each bullet to 1-2 sentences. Use cricket terminology. Be specific to these teams. Do NOT use markdown headers. Use plain bullet points with • character."""


def generate_match_analysis(match_context: dict) -> str | None:
    """Call Gemini API to generate pre-match analysis.

    Returns the analysis text, or None if the API call fails or no key is configured.
    """
    settings = get_settings()
    api_key = settings.gemini_api_key
    if not api_key:
        log.warning("Gemini API key not configured — skipping analysis.")
        return None

    model = settings.gemini_model
    url = _API_URL.format(model=model)
    log.info("Gemini: calling %s for %s vs %s", model,
             match_context.get("team_a"), match_context.get("team_b"))

    prompt = _build_prompt(match_context)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 512,
            "topP": 0.9,
        },
    }

    try:
        resp = httpx.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=_TIMEOUT,
            verify=_get_ssl_context(),
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract text from Gemini response
        candidates = data.get("candidates", [])
        if not candidates:
            log.warning("Gemini returned no candidates.")
            return None

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            log.warning("Gemini returned empty text.")
            return None

        log.info("Gemini: got %d chars analysis for %s vs %s",
                 len(text), match_context.get("team_a"), match_context.get("team_b"))
        return text

    except httpx.HTTPStatusError as exc:
        log.warning("Gemini API error %s: %s", exc.response.status_code, exc.response.text[:200])
        return None
    except Exception as exc:
        log.warning("Gemini call failed: %s", exc)
        return None
