"""Homepage router — serves the IPL 2026 live prediction dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter()

_STATUS_ICON = {1: "✅", 0: "❌", None: "⏳"}
_STATUS_LABEL = {1: "Correct", 0: "Wrong", None: "Pending"}
_STATUS_BG = {1: "bg-green-50", 0: "bg-red-50", None: "bg-yellow-50"}
_STATUS_BADGE = {
    1: "bg-green-100 text-green-800",
    0: "bg-red-100 text-red-800",
    None: "bg-yellow-100 text-yellow-800",
}


def _render_homepage(next_pred: dict | None, history: list[dict], stats: dict, upcoming: list[dict], overrides: list[dict]) -> str:
    accuracy = stats.get("accuracy_pct", 0.0)
    total = stats.get("total", 0)
    correct = stats.get("correct", 0)
    wrong_since = stats.get("wrong_since_retrain", 0)
    last_retrain = stats.get("last_retrain_at") or "Never"

    # ── Next match section ──────────────────────────────────
    if next_pred and next_pred.get("predicted_winner"):
        team_a = next_pred.get("team_a", "TBA")
        team_b = next_pred.get("team_b", "TBA")
        venue = next_pred.get("venue", "TBA")
        match_date = next_pred.get("match_date", "")
        winner = next_pred.get("predicted_winner", "TBA")
        ta_pct = round(next_pred.get("team_a_probability", 0.5) * 100, 1)
        tb_pct = round(100 - ta_pct, 1)
        explanation = next_pred.get("explanation", "")
        ai_analysis = next_pred.get("ai_analysis", "") or ""
        # Strip toss-related lines — toss is unknown before match starts
        explanation = "; ".join(
            part for part in explanation.split("; ")
            if "toss" not in part.lower()
        )
        winner_pct = ta_pct if winner == team_a else tb_pct

        next_html = f"""
        <div class="bg-white rounded-2xl shadow-lg overflow-hidden border border-indigo-100">
          <div class="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4">
            <div class="flex items-center justify-between">
              <span class="text-white font-semibold text-lg">🏏 Next Match Prediction</span>
              <span class="bg-white/20 text-white text-sm px-3 py-1 rounded-full">{match_date}</span>
            </div>
          </div>
          <div class="px-6 py-5">
            <div class="flex items-center justify-center gap-4 mb-5">
              <span class="text-2xl font-bold text-gray-800">{team_a}</span>
              <span class="text-gray-400 font-medium">vs</span>
              <span class="text-2xl font-bold text-gray-800">{team_b}</span>
            </div>
            <p class="text-center text-sm text-gray-500 mb-4">📍 {venue}</p>
            <!-- Probability bar -->
            <div class="mb-3">
              <div class="flex justify-between text-xs text-gray-500 mb-1">
                <span>{team_a} {ta_pct}%</span>
                <span>{team_b} {tb_pct}%</span>
              </div>
              <div class="flex h-4 rounded-full overflow-hidden">
                <div class="bg-indigo-500 transition-all" style="width:{ta_pct}%"></div>
                <div class="bg-purple-400 flex-1"></div>
              </div>
            </div>
            <div class="bg-indigo-50 rounded-xl p-4 text-center mt-4">
              <p class="text-xs text-indigo-500 mb-1 uppercase tracking-wide font-semibold">Pre-match Prediction</p>
              <p class="text-3xl font-extrabold text-indigo-700">{winner}</p>
              <p class="text-indigo-500 text-sm mt-1">{winner_pct}% win probability</p>
              <p class="text-indigo-300 text-xs mt-1">Toss &amp; pitch conditions not yet known · prediction updates automatically</p>
            </div>
            {f'<p class="text-gray-500 text-xs mt-3 text-center italic">{explanation}</p>' if explanation else ""}
          </div>
        </div>"""

        # AI Analysis panel (Gemini)
        if ai_analysis:
            # Convert bullet points to HTML list items
            lines = [ln.strip() for ln in ai_analysis.splitlines() if ln.strip()]
            li_items = ""
            for ln in lines:
                # Strip leading bullet characters (•, -, *, numbered)
                clean = ln.lstrip("•-*0123456789.) ").strip()
                if clean:
                    li_items += f'<li class="text-gray-700 text-sm leading-relaxed">{clean}</li>\n'

            next_html += f"""
        <div class="bg-white rounded-2xl shadow border border-gray-100 overflow-hidden mt-4">
          <div class="px-6 py-4 border-b border-gray-100 flex items-center gap-2">
            <span class="text-lg">🤖</span>
            <h2 class="font-semibold text-gray-700">AI Match Analysis</h2>
            <span class="ml-auto text-[10px] bg-purple-100 text-purple-600 px-2 py-0.5 rounded-full font-medium">Powered by Gemini</span>
          </div>
          <div class="px-6 py-5">
            <ul class="space-y-3 list-disc list-inside">
              {li_items}
            </ul>
          </div>
        </div>"""
    else:
        next_html = """
        <div class="bg-white rounded-2xl shadow border border-gray-100 p-8 text-center text-gray-500">
          <p class="text-4xl mb-2">🏟️</p>
          <p class="font-medium">No upcoming match found — check back soon</p>
        </div>"""

    # ── Upcoming match table ─────────────────────────────────
    from datetime import date
    today = date.today().isoformat()
    future_matches = [m for m in upcoming if m.get("match_date", "") > today][:7]

    upcoming_rows = ""
    for idx, m in enumerate(future_matches):
        u_ta = m.get("team_a", "")
        u_tb = m.get("team_b", "")
        u_venue = m.get("venue", "")
        u_date = m.get("match_date", "")
        u_winner = m.get("predicted_winner") or "—"
        u_ta_pct = round((m.get("team_a_probability") or 0.5) * 100, 1)
        u_tb_pct = round(100 - u_ta_pct, 1)
        u_analysis = (m.get("ai_analysis") or "").strip()
        winner_cls = "text-indigo-700 font-semibold" if u_winner == u_ta else "text-purple-700 font-semibold"
        bar_a = int(u_ta_pct)
        analysis_btn = ""
        analysis_row = ""
        if u_analysis:
            analysis_btn = f'<button onclick="document.getElementById(\'ai-{idx}\').classList.toggle(\'hidden\')" class="ml-2 text-[10px] bg-purple-50 text-purple-600 px-2 py-0.5 rounded-full hover:bg-purple-100 transition cursor-pointer">🤖 AI</button>'
            # Convert to short HTML
            a_lines = [ln.lstrip("•-*0123456789.) ").strip() for ln in u_analysis.splitlines() if ln.strip()]
            a_html = " ".join(f"<span>• {l}</span>" for l in a_lines if l)
            analysis_row = f'<tr id="ai-{idx}" class="hidden bg-purple-50/50"><td colspan="5" class="px-4 py-3 text-xs text-gray-600 leading-relaxed">{a_html}</td></tr>'
        upcoming_rows += f"""
        <tr class="border-b border-gray-100 hover:bg-gray-50 transition">
          <td class="px-4 py-3 text-sm text-gray-500 whitespace-nowrap">{u_date}</td>
          <td class="px-4 py-3 text-sm font-medium text-gray-800 whitespace-nowrap">{u_ta} <span class="text-gray-400 font-normal">vs</span> {u_tb}{analysis_btn}</td>
          <td class="px-4 py-3 text-sm text-gray-500 whitespace-nowrap hidden md:table-cell">{u_venue}</td>
          <td class="px-4 py-3">
            <div class="flex h-2 rounded-full overflow-hidden w-24">
              <div class="bg-indigo-400" style="width:{bar_a}%"></div>
              <div class="bg-purple-300 flex-1"></div>
            </div>
            <div class="flex justify-between text-[10px] text-gray-400 mt-0.5 w-24">
              <span>{u_ta_pct}%</span><span>{u_tb_pct}%</span>
            </div>
          </td>
          <td class="px-4 py-3 text-sm {winner_cls} whitespace-nowrap">{u_winner}</td>
        </tr>{analysis_row}"""

    upcoming_section = f"""
    <div class="bg-white rounded-2xl shadow border border-gray-100 overflow-hidden">
      <div class="px-6 py-4 border-b border-gray-100">
        <h2 class="font-semibold text-gray-700">📅 Upcoming Matches &amp; Predictions</h2>
        <p class="text-xs text-gray-400 mt-0.5">Next 7 fixtures · pre-match win probability · 7:30 PM IST</p>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left">
          <thead>
            <tr class="bg-gray-50 text-xs text-gray-500 uppercase tracking-wider">
              <th class="px-4 py-3 font-medium">Date</th>
              <th class="px-4 py-3 font-medium">Match</th>
              <th class="px-4 py-3 font-medium hidden md:table-cell">Venue</th>
              <th class="px-4 py-3 font-medium">Probability</th>
              <th class="px-4 py-3 font-medium">Predicted Winner</th>
            </tr>
          </thead>
          <tbody>
            {upcoming_rows if upcoming_rows else '<tr><td colspan="5" class="px-4 py-6 text-center text-gray-400">No upcoming fixtures found</td></tr>'}
          </tbody>
        </table>
      </div>
    </div>""" if future_matches else ""

    # ── Override badges (active notes) ────────────────────────────────
    override_badges = ""
    for ov in overrides:
        p = ov.get("parsed", {})
        desc = p.get("description", ov.get("note", ""))
        ov_id = ov.get("id", "")
        bowl_d = p.get("bowl_delta", 0.0)
        bat_d  = p.get("bat_delta",  0.0)
        tag_cls = "bg-red-50 border-red-200 text-red-800" if (bowl_d < 0 or bat_d < 0) else "bg-blue-50 border-blue-200 text-blue-800"
        override_badges += f"""
        <div class="flex items-center justify-between gap-3 px-4 py-2 rounded-lg border {tag_cls} text-sm">
          <span>{desc}</span>
          <a href="/override/delete/{ov_id}" class="text-xs opacity-60 hover:opacity-100 shrink-0" title="Remove">✕</a>
        </div>"""

    if not override_badges:
        override_badges = '<p class="text-gray-400 text-sm px-1">No active overrides — predictions use live standings data.</p>'

    overrides_section = f"""
    <div class="bg-white rounded-2xl shadow border border-gray-100 overflow-hidden">
      <div class="px-6 py-4 border-b border-gray-100">
        <h2 class="font-semibold text-gray-700">🧠 Match Context &amp; Overrides</h2>
        <p class="text-xs text-gray-400 mt-0.5">Tell the model about injuries, pitch reports, team news · predictions auto-regenerate</p>
      </div>
      <div class="p-5 space-y-4">
        <!-- Active overrides -->
        <div class="space-y-2">
          {override_badges}
        </div>
        <!-- Input form -->
        <form method="POST" action="/override" class="space-y-3">
          <textarea
            name="note"
            rows="3"
            placeholder="e.g. Pat Cummins is injured&#10;KKR missing Varun Chakravarthy&#10;Eden Gardens is a batting pitch today"
            class="w-full border border-gray-200 rounded-xl px-4 py-3 text-sm text-gray-700 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-400 resize-none"
          ></textarea>
          <div class="flex items-center gap-3">
            <button type="submit"
              class="bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold px-5 py-2 rounded-lg transition">
              Apply &amp; Re-predict
            </button>
            <a href="/override/clear" class="text-sm text-gray-400 hover:text-red-500 transition">Clear all overrides</a>
          </div>
        </form>
      </div>
    </div>"""
    rows_html = ""
    if not history:
        rows_html = '<tr><td colspan="6" class="px-4 py-6 text-center text-gray-400">No predictions recorded yet</td></tr>'
    for rec in history:
        ic = rec.get("is_correct")
        icon = _STATUS_ICON[ic]
        label = _STATUS_LABEL[ic]
        row_bg = _STATUS_BG[ic]
        badge_cls = _STATUS_BADGE[ic]
        ta = rec.get("team_a", "")
        tb = rec.get("team_b", "")
        pw = rec.get("predicted_winner", "")
        aw = rec.get("actual_winner") or "—"
        ta_p = round((rec.get("team_a_probability") or 0.5) * 100, 0)
        tb_p = round(100 - ta_p, 0)
        bar_a = int(ta_p)
        rows_html += f"""
        <tr class="{row_bg} border-b border-gray-100 hover:bg-opacity-80 transition">
          <td class="px-4 py-3 text-sm font-medium text-gray-700 whitespace-nowrap">{rec.get("match_date","")}</td>
          <td class="px-4 py-3 text-sm text-gray-800">{ta} vs {tb}</td>
          <td class="px-4 py-3">
            <div class="flex h-2 rounded-full overflow-hidden w-24">
              <div class="bg-indigo-400" style="width:{bar_a}%"></div>
              <div class="bg-purple-300 flex-1"></div>
            </div>
            <div class="flex justify-between text-xs text-gray-400 mt-0.5 w-24">
              <span>{int(ta_p)}%</span><span>{int(tb_p)}%</span>
            </div>
          </td>
          <td class="px-4 py-3 text-sm font-semibold text-indigo-700">{pw}</td>
          <td class="px-4 py-3 text-sm text-gray-600">{aw}</td>
          <td class="px-4 py-3">
            <span class="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium {badge_cls}">
              {icon} {label}
            </span>
          </td>
        </tr>"""

    # ── Accuracy ring (CSS only) ───────────────────────────
    ring_pct = int(accuracy)
    # stroke-dashoffset: 440 = full circle, reduce by accuracy%
    dash_offset = 440 - int(4.40 * ring_pct)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IPL 2026 Cricket Predictor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <meta http-equiv="refresh" content="300">
  <style>
    .ring-circle {{ stroke-dasharray: 440; stroke-dashoffset: {dash_offset}; transition: stroke-dashoffset 1s; }}
  </style>
</head>
<body class="bg-gray-50 min-h-screen font-sans">

  <!-- Header -->
  <header class="bg-gradient-to-r from-indigo-700 via-purple-700 to-indigo-800 shadow-lg">
    <div class="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <span class="text-3xl">🏏</span>
        <div>
          <h1 class="text-white text-xl font-bold">Cricket Predictor</h1>
          <p class="text-indigo-200 text-xs">IPL 2026 · AI-powered · Self-learning · Gemini Analysis</p>
        </div>
      </div>
      <span class="text-indigo-200 text-xs">Auto-refresh every 5 min</span>
    </div>
  </header>

  <main class="max-w-6xl mx-auto px-4 py-8 space-y-8">

    <!-- Stats strip -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div class="bg-white rounded-xl shadow p-4 text-center">
        <p class="text-3xl font-extrabold text-indigo-600">{total}</p>
        <p class="text-xs text-gray-500 mt-1">Predictions Made</p>
      </div>
      <div class="bg-white rounded-xl shadow p-4 text-center">
        <p class="text-3xl font-extrabold text-green-600">{correct}</p>
        <p class="text-xs text-gray-500 mt-1">Correct</p>
      </div>
      <div class="bg-white rounded-xl shadow p-4 text-center">
        <div class="relative inline-flex items-center justify-center">
          <svg class="w-16 h-16 -rotate-90" viewBox="0 0 160 160">
            <circle cx="80" cy="80" r="70" fill="none" stroke="#e5e7eb" stroke-width="16"/>
            <circle cx="80" cy="80" r="70" fill="none" stroke="#4f46e5" stroke-width="16"
              class="ring-circle"/>
          </svg>
          <span class="absolute text-lg font-bold text-indigo-700">{ring_pct}%</span>
        </div>
        <p class="text-xs text-gray-500 mt-1">Accuracy</p>
      </div>
      <div class="bg-white rounded-xl shadow p-4 text-center">
        <p class="text-3xl font-extrabold {'text-red-500' if wrong_since > 3 else 'text-gray-700'}">{wrong_since}</p>
        <p class="text-xs text-gray-500 mt-1">Wrong since retrain</p>
        <p class="text-xs text-gray-400 mt-0.5">Last: {last_retrain[:10] if last_retrain != 'Never' else 'Never'}</p>
      </div>
    </div>

    <!-- Next match prediction (full width) -->
    <div class="max-w-2xl mx-auto">
      {next_html}
    </div>

    {upcoming_section}

    {overrides_section}

    <!-- Prediction history -->
    <div class="bg-white rounded-2xl shadow border border-gray-100 overflow-hidden">
      <div class="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
        <h2 class="font-semibold text-gray-700">📜 Prediction History (last 10)</h2>
        <span class="text-xs text-gray-400">✅ Correct &nbsp; ❌ Wrong &nbsp; ⏳ Pending</span>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left">
          <thead class="bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
            <tr>
              <th class="px-4 py-3">Date</th>
              <th class="px-4 py-3">Match</th>
              <th class="px-4 py-3">Win %</th>
              <th class="px-4 py-3">Predicted</th>
              <th class="px-4 py-3">Actual</th>
              <th class="px-4 py-3">Result</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
    </div>

  </main>

  <footer class="text-center text-xs text-gray-400 py-6">
    Cricket Predictor · Powered by CricSheet, Cricmetric &amp; Gemini AI · Deployed on Render
  </footer>

</body>
</html>"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def homepage() -> HTMLResponse:
    from cricket_predictor.services.prediction_tracker import get_prediction_tracker

    tracker = get_prediction_tracker()
    next_pred = tracker.get_next_match_prediction()
    history = tracker.get_recent_history(10)
    stats = tracker.get_accuracy_stats()
    upcoming = tracker._db.get_upcoming_predictions(7)
    overrides = tracker.get_active_overrides()
    html = _render_homepage(next_pred, history, stats, upcoming, overrides)
    return HTMLResponse(content=html)


@router.post("/override", include_in_schema=False)
async def add_override(note: str = Form(...)) -> RedirectResponse:
    from cricket_predictor.services.prediction_tracker import get_prediction_tracker

    if note.strip():
        tracker = get_prediction_tracker()
        tracker.add_override(note.strip())
    return RedirectResponse(url="/", status_code=303)


@router.get("/override/delete/{override_id}", include_in_schema=False)
async def delete_override(override_id: int) -> RedirectResponse:
    from cricket_predictor.services.prediction_tracker import get_prediction_tracker

    get_prediction_tracker().delete_override(override_id)
    return RedirectResponse(url="/", status_code=303)


@router.get("/override/clear", include_in_schema=False)
async def clear_overrides() -> RedirectResponse:
    from cricket_predictor.services.prediction_tracker import get_prediction_tracker

    tracker = get_prediction_tracker()
    for ov in tracker.get_active_overrides():
        tracker.delete_override(ov["id"])
    return RedirectResponse(url="/", status_code=303)


@router.get("/admin/regenerate", include_in_schema=False)
async def admin_regenerate() -> RedirectResponse:
    """Clear all future predictions, refresh injuries, and re-predict."""
    from cricket_predictor.services.prediction_tracker import get_prediction_tracker
    from cricket_predictor.services.standings_service import get_standings_service

    tracker = get_prediction_tracker()
    # Refresh standings
    try:
        await get_standings_service().refresh()
    except Exception:
        pass
    # Refresh injury overrides (clears stale, fetches fresh)
    try:
        tracker.refresh_injury_overrides()
    except Exception:
        pass
    # Force-clear ALL future predictions so they are re-generated with Gemini
    tracker._invalidate_future_predictions()
    # Regenerate all predictions with current data + Gemini analysis
    tracker.predict_upcoming_matches()
    return RedirectResponse(url="/", status_code=303)


@router.get("/admin/debug", include_in_schema=False)
async def admin_debug() -> dict:
    """Debug endpoint showing config state and prediction diagnostics."""
    from cricket_predictor.config.settings import get_settings
    from cricket_predictor.services.prediction_tracker import get_prediction_tracker

    settings = get_settings()
    tracker = get_prediction_tracker()
    upcoming = tracker._db.get_upcoming_predictions(7)
    overrides = tracker.get_active_overrides()

    return {
        "gemini_key_set": bool(settings.gemini_api_key),
        "gemini_key_prefix": settings.gemini_api_key[:10] + "..." if settings.gemini_api_key else "(empty)",
        "gemini_model": settings.gemini_model,
        "upcoming_count": len(upcoming),
        "overrides_count": len(overrides),
        "predictions": [
            {
                "match_id": p.get("match_id"),
                "teams": f"{p.get('team_a')} vs {p.get('team_b')}",
                "winner": p.get("predicted_winner"),
                "ta_prob": round((p.get("team_a_probability") or 0.5) * 100, 1),
                "has_ai_analysis": bool(p.get("ai_analysis")),
                "ai_analysis_preview": (p.get("ai_analysis") or "")[:120],
            }
            for p in upcoming
        ],
    }
