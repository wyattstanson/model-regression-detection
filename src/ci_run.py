"""
ci_run.py — CI/CD entrypoint.

Called by GitHub Actions on every PR that modifies /prompts.
Exit codes:
  0 = ok or warn (merge allowed)
  1 = critical regression (merge blocked)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    from src.prompt_loader import load_prompt
    from src.eval_runner import run_eval, load_recent_runs
    from src.diff_engine import diff_runs, get_baseline_run
    from src.reporter import generate_html_report, send_slack_alert, build_pr_comment

  
    version = os.getenv("PROMPT_VERSION")
    logger.info(f"Loading prompt: {version or 'latest'}")
    config = load_prompt(version)
    logger.info(f"Prompt v{config.version} loaded ({len(config.few_shot_examples)} few-shot examples)")

  
    use_judge = os.getenv("USE_JUDGE", "true").lower() == "true"
    current = run_eval(config, use_judge=use_judge)

    
    baseline = get_baseline_run()
    if baseline is None:
        logger.info("No baseline available — first run. Saving as baseline.")
        print(f"\n✓ First eval run complete: {current.pass_rate:.1%} pass rate")
        print(f"  Run ID: {current.run_id}")
        print(f"  This run is now the baseline for future comparisons.")
        return 0

    diff = diff_runs(current, baseline)

    report_path = generate_html_report(current, diff)

    if diff.status in ("warn", "critical"):
        send_slack_alert(diff, current, report_path)

    comment = build_pr_comment(diff, current)
    print("\n" + "─" * 60)
    print("PR COMMENT PAYLOAD:")
    print("─" * 60)
    print(comment)
    print("─" * 60)

  
    print(f"\n{diff.summary_line()}")
    print(f"  Report: {report_path}")

    if diff.status == "critical":
        print("\n🚨 MERGE BLOCKED — regression delta exceeds critical threshold.")
        return 1

    if diff.status == "warn":
        print("\n⚠️  Warning — regression detected but within tolerable range.")

    return 0


if __name__ == "__main__":
    sys.exit(main())