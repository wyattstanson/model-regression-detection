"""
reporter.py — Alerting and reporting layer.

Generates:
  1. HTML diff report (saved to /reports/)
  2. Slack webhook alert
  3. Plain-text PR comment payload
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from src.diff_engine import DiffResult, CaseFlip
from src.eval_runner import EvalRun

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"




def generate_html_report(current: EvalRun, diff: DiffResult) -> Path:
    """Generate an HTML diff report and return its path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"report_{current.run_id}.html"

    status_color = {"ok": "#22c55e", "warn": "#f59e0b", "critical": "#ef4444"}[diff.status]
    delta_sign   = "+" if diff.delta >= 0 else ""

    regression_rows = _render_flip_rows(diff.regressions, "regression")
    improvement_rows = _render_flip_rows(diff.improvements, "improvement")

    cat_rows = ""
    for cat, d in diff.per_category_delta.items():
        color = "#22c55e" if d >= 0 else "#ef4444"
        cat_rows += f"<tr><td>{cat}</td><td style='color:{color}'>{d:+.1%}</td></tr>"

    slow_drift_banner = ""
    if diff.slow_drift_warning:
        slow_drift_banner = f"""
        <div class="banner warn">
          ⚠ Slow drift detected — {diff.moving_avg_pass_rate:.1%} rolling average
          (last 7 runs). No single run triggered an alert, but gradual degradation
          is accumulating.
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Eval Report — {current.run_id}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 24px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  .meta {{ color: #94a3b8; font-size: 0.85rem; margin-bottom: 24px; }}
  .status-badge {{ display:inline-block; padding: 4px 12px; border-radius: 9999px;
                   background: {status_color}22; color: {status_color};
                   font-weight: 600; font-size: 0.9rem; border: 1px solid {status_color}; }}
  .scorecard {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 16px; }}
  .card .label {{ color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .card .value {{ font-size: 1.8rem; font-weight: 700; margin-top: 4px; }}
  .card .delta {{ font-size: 0.85rem; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 0.85rem; }}
  th {{ text-align: left; padding: 8px 12px; background: #1e293b; color: #94a3b8; font-weight: 500; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #1e293b; }}
  .tag-reg  {{ background: #ef444422; color: #ef4444; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }}
  .tag-imp  {{ background: #22c55e22; color: #22c55e; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }}
  .section {{ margin-top: 32px; }}
  .banner {{ padding: 12px 16px; border-radius: 8px; margin: 16px 0; }}
  .banner.warn {{ background: #f59e0b22; border: 1px solid #f59e0b; color: #fbbf24; }}
  h2 {{ font-size: 1rem; font-weight: 600; margin-bottom: 8px; color: #cbd5e1; }}
</style>
</head>
<body>
<h1>Eval Report <span class="status-badge">{diff.status.upper()}</span></h1>
<div class="meta">
  Run ID: {current.run_id} &nbsp;·&nbsp;
  Prompt: v{current.prompt_version} &nbsp;·&nbsp;
  Model: {current.model} &nbsp;·&nbsp;
  {current.timestamp[:19].replace("T", " ")} UTC
</div>

{slow_drift_banner}

<div class="scorecard">
  <div class="card">
    <div class="label">Pass rate</div>
    <div class="value">{current.pass_rate:.1%}</div>
    <div class="delta" style="color:{status_color}">{delta_sign}{diff.delta:.1%} vs baseline</div>
  </div>
  <div class="card">
    <div class="label">Regressions</div>
    <div class="value" style="color:{'#ef4444' if diff.regression_count else '#22c55e'}">{diff.regression_count}</div>
    <div class="delta" style="color:#94a3b8">cases flipped pass→fail</div>
  </div>
  <div class="card">
    <div class="label">Improvements</div>
    <div class="value" style="color:#22c55e">{diff.improvement_count}</div>
    <div class="delta" style="color:#94a3b8">cases flipped fail→pass</div>
  </div>
  <div class="card">
    <div class="label">Avg latency</div>
    <div class="value">{current.avg_latency_ms:.0f}<span style="font-size:1rem;font-weight:400">ms</span></div>
    <div class="delta" style="color:#94a3b8">{current.total_tokens:,} total tokens</div>
  </div>
</div>

<div class="section">
  <h2>Per-category accuracy delta</h2>
  <table>
    <tr><th>Category</th><th>Delta vs baseline</th></tr>
    {cat_rows}
  </table>
</div>

{f'<div class="section"><h2>Regressions ({diff.regression_count})</h2><table><tr><th>Case ID</th><th>Expected</th><th>Old prediction</th><th>New prediction</th><th>Status</th></tr>{regression_rows}</table></div>' if diff.regressions else ''}

{f'<div class="section"><h2>Improvements ({diff.improvement_count})</h2><table><tr><th>Case ID</th><th>Expected</th><th>Old prediction</th><th>New prediction</th><th>Status</th></tr>{improvement_rows}</table></div>' if diff.improvements else ''}

</body>
</html>"""

    path.write_text(html)
    logger.info(f"HTML report written to {path}")
    return path


def _render_flip_rows(flips: list[CaseFlip], kind: str) -> str:
    tag_class = "tag-reg" if kind == "regression" else "tag-imp"
    label = "REGRESSION" if kind == "regression" else "IMPROVEMENT"
    rows = ""
    for flip in flips:
        rows += (
            f"<tr>"
            f"<td><code>{flip.case_id}</code></td>"
            f"<td>{flip.expected_category}</td>"
            f"<td>{flip.old_category or '—'}</td>"
            f"<td>{flip.new_category or '—'}</td>"
            f"<td><span class='{tag_class}'>{label}</span></td>"
            f"</tr>"
        )
    return rows




def send_slack_alert(diff: DiffResult, current: EvalRun, report_path: Path) -> bool:
    """
    Send structured Slack alert via incoming webhook.
    Returns True on success.
    Reads SLACK_WEBHOOK_URL from environment.
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set — skipping Slack alert")
        return False

    status_emoji = {"ok": "✅", "warn": "⚠️", "critical": "🚨"}[diff.status]
    delta_sign = "+" if diff.delta >= 0 else ""

    drift_note = ""
    if diff.slow_drift_warning:
        drift_note = f"\n *Slow drift*: {diff.moving_avg_pass_rate:.1%} rolling avg (7 runs)"

    payload = {
        "text": f"{status_emoji} *Eval {diff.status.upper()}* — prompt v{current.prompt_version}",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{status_emoji} Eval {diff.status.upper()} — {current.prompt_version}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Pass rate*\n{current.pass_rate:.1%} ({delta_sign}{diff.delta:.1%})"},
                    {"type": "mrkdwn", "text": f"*Regressions*\n{diff.regression_count} cases"},
                    {"type": "mrkdwn", "text": f"*Improvements*\n{diff.improvement_count} cases"},
                    {"type": "mrkdwn", "text": f"*Tokens used*\n{current.total_tokens:,}"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"Accuracy dropped from *{diff.baseline_pass_rate:.1%}* to *{diff.current_pass_rate:.1%}*"
                        f"{drift_note}"
                    ),
                },
            },
        ],
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            success = resp.status == 200
            logger.info(f"Slack alert sent: status={resp.status}")
            return success
    except Exception as e:
        logger.error(f"Slack alert failed: {e}")
        return False




def build_pr_comment(diff: DiffResult, current: EvalRun) -> str:
    """Return a Markdown string suitable for posting as a GitHub PR comment."""
    status_emoji = {"ok": "✅", "warn": "⚠️", "critical": "🚨"}[diff.status]
    delta_sign = "+" if diff.delta >= 0 else ""

    lines = [
        f"## {status_emoji} Eval {diff.status.upper()} — prompt v{current.prompt_version}",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Pass rate | {current.pass_rate:.1%} ({delta_sign}{diff.delta:.1%} vs baseline) |",
        f"| Regressions | {diff.regression_count} |",
        f"| Improvements | {diff.improvement_count} |",
        f"| Avg latency | {current.avg_latency_ms:.0f}ms |",
        f"| Total tokens | {current.total_tokens:,} |",
        "",
    ]

    if diff.slow_drift_warning:
        lines += [
            f">  **Slow drift**: {diff.moving_avg_pass_rate:.1%} rolling average over last 7 runs.",
            "",
        ]

    if diff.regressions:
        lines += ["### Regressions", ""]
        for flip in diff.regressions:
            lines.append(
                f"- `{flip.case_id}`: expected `{flip.expected_category}`, "
                f"was `{flip.old_category}`, now `{flip.new_category}`"
            )
        lines.append("")

    if diff.status == "critical":
        lines.append("> 🚨 **Merge blocked** — regression delta exceeds critical threshold.")

    return "\n".join(lines)