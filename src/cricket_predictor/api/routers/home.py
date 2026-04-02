"""Homepage router — serves the IPL 2026 live prediction dashboard."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_STATUS_ICON = {1: "✅", 0: "❌", None: "⏳"}
_STATUS_LABEL = {1: "Correct", 0: "Wrong", None: "Pending"}
_STATUS_BG = {1: "bg-green-50", 0: "bg-red-50", None: "bg-yellow-50"}
_STATUS_BADGE = {
    1: "bg-green-100 text-green-800",
    0: "bg-red-100 text-red-800",
    None: "bg-yellow-100 text-yellow-800",
}


def _render_homepage(next_pred: dict | None, history: list[dict], stats: dict) -> str:
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
        conf = round(next_pred.get("confidence_score", 0) * 100, 0)
        explanation = next_pred.get("explanation", "")
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
              <p class="text-xs text-indigo-500 mb-1 uppercase tracking-wide font-semibold">Predicted Winner</p>
              <p class="text-3xl font-extrabold text-indigo-700">{winner}</p>
              <p class="text-indigo-500 text-sm mt-1">{winner_pct}% probability · {conf:.0f}% confidence</p>
            </div>
            {f'<p class="text-gray-500 text-xs mt-3 text-center italic">{explanation}</p>' if explanation else ""}
          </div>
        </div>"""
    else:
        next_html = """
        <div class="bg-white rounded-2xl shadow border border-gray-100 p-8 text-center text-gray-500">
          <p class="text-4xl mb-2">🏟️</p>
          <p class="font-medium">No upcoming match found — check back soon</p>
        </div>"""

    # ── History rows ──────────────────────────────────────
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
          <p class="text-indigo-200 text-xs">IPL 2026 · AI-powered · Self-learning</p>
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

    <!-- Two-column layout -->
    <div class="grid md:grid-cols-2 gap-6 items-start">

      <!-- Next match prediction -->
      {next_html}

      <!-- Quick links -->
      <div class="bg-white rounded-2xl shadow border border-gray-100 p-5 space-y-3">
        <h2 class="font-semibold text-gray-700 mb-3">🔗 API Endpoints</h2>
        {''.join(f'''
        <a href="{url}" target="_blank"
           class="flex items-center gap-3 p-3 rounded-xl hover:bg-indigo-50 transition group">
          <span class="bg-indigo-100 text-indigo-600 rounded-lg p-2 text-sm group-hover:bg-indigo-200">{icon}</span>
          <div>
            <p class="text-sm font-medium text-gray-700">{label}</p>
            <p class="text-xs text-gray-400">{url}</p>
          </div>
        </a>''' for icon, label, url in [
            ("📊", "Match Prediction", "/predict/match"),
            ("🧑", "Player Prediction", "/predict/player"),
            ("🏆", "IPL Standings", "/standings"),
            ("🔄", "Refresh Standings", "/standings/refresh"),
            ("📋", "API Docs", "/docs"),
            ("❤️", "Health Check", "/health"),
        ])}
      </div>
    </div>

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
    Cricket Predictor · Powered by CricSheet &amp; Cricmetric · Deployed on Render
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
    html = _render_homepage(next_pred, history, stats)
    return HTMLResponse(content=html)
