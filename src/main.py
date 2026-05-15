
"""
CLI entrypoint for the Model Regression Detection System.

Usage
-----
  # Run eval against latest prompt version, diff vs baseline:
  python main.py run

  # Run against a specific version:
  python main.py run --version 1.1.0

  # List available prompt versions:
  python main.py list-prompts

  # Show last N eval runs:
  python main.py list-runs --limit 5

  # Print a diff between two run IDs:
  python main.py diff <current_run_id> <baseline_run_id>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from src.prompt_loader import list_prompts, load_prompt
from src.eval_runner   import list_runs, load_run, run_eval
from src.diff_engine   import diff_runs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")




def cmd_run(args: argparse.Namespace) -> int:
    version = args.version or "latest"
    logger.info("Loading prompt version: %s", version)
    config = load_prompt(version)
    logger.info("Loaded: %s (%s)", config.name, config.version)

    def progress(done: int, total: int) -> None:
        pct = done / total * 100
        print(f"\r  Progress: {done}/{total}  ({pct:.0f}%)", end="", flush=True)

    run = run_eval(config, progress_cb=progress)
    print()  

    print(f"\n{'='*60}")
    print(f"  Run ID  : {run.run_id}")
    print(f"  Version : {run.prompt_version}")
    print(f"  Accuracy: {(run.accuracy or 0)*100:.1f}%")
    print(f"  Latency : {run.avg_latency_ms:.0f} ms avg")
    print(f"  Cost    : ${run.total_cost_usd:.4f}")
    print(f"{'='*60}\n")

    if args.baseline:
        baseline = load_run(args.baseline)
        if baseline is None:
            logger.error("Baseline run '%s' not found in database.", args.baseline)
            return 1
        diff = diff_runs(current=run, baseline=baseline)
        _print_diff(diff)

    return 0


def cmd_list_prompts(_args: argparse.Namespace) -> int:
    versions = list_prompts()
    if not versions:
        print("No prompt versions found in ./prompts/")
        return 0
    print("Available prompt versions:")
    for v in versions:
        marker = "  (latest)" if v == versions[-1] else ""
        print(f"  v{v}{marker}")
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    runs = list_runs()
    if not runs:
        print("No eval runs found in the database.")
        return 0
    limit = args.limit
    print(f"{'RUN ID':<36}  {'VERSION':<8}  {'ACCURACY':>8}  {'LATENCY':>9}  {'COST':>8}  STATUS")
    print("-" * 85)
    for row in runs[:limit]:
        acc  = f"{(row['accuracy'] or 0)*100:.1f}%" if row['accuracy'] is not None else "—"
        lat  = f"{row['avg_latency_ms']:.0f}ms"    if row['avg_latency_ms'] is not None else "—"
        cost = f"${row['total_cost_usd']:.4f}"      if row['total_cost_usd'] is not None else "—"
        print(f"{row['run_id']:<36}  {row['prompt_version']:<8}  {acc:>8}  {lat:>9}  {cost:>8}  {row['status']}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    current  = load_run(args.current_run_id)
    baseline = load_run(args.baseline_run_id)

    if current is None:
        logger.error("Current run '%s' not found.", args.current_run_id)
        return 1
    if baseline is None:
        logger.error("Baseline run '%s' not found.", args.baseline_run_id)
        return 1

    diff = diff_runs(current=current, baseline=baseline)
    _print_diff(diff)
    return 0




def _print_diff(diff) -> None:
    print(f"\n{'='*60}")
    print(f"  DIFF  {diff.baseline_version} → {diff.current_version}")
    print(f"{'='*60}")
    delta_sign = lambda x: (f"+{x:.1%}" if x and x >= 0 else f"{x:.1%}") if x is not None else "—"
    print(f"  Accuracy delta  : {delta_sign(diff.accuracy_delta)}")
    print(f"  Latency delta   : {diff.latency_delta_ms:+.0f}ms" if diff.latency_delta_ms is not None else "  Latency delta   : —")
    print(f"  Regressions     : {len(diff.regressions)}")
    print(f"  Improvements    : {len(diff.improvements)}")
    print(f"  Neutral changes : {len(diff.neutral_changes)}")
    print(f"  Slow-drift cases: {len(diff.slow_drift_cases)}")

    if diff.regressions:
        print("\n  ⚠️  Regressions (correct→wrong):")
        for r in diff.regressions:
            print(f"    [{r.case_id}]  {r.baseline_label} → {r.current_label}  conf_delta={r.confidence_delta}")

    if diff.improvements:
        print("\n  ✅ Improvements (wrong→correct):")
        for r in diff.improvements:
            print(f"    [{r.case_id}]  {r.baseline_label} → {r.current_label}")

    verdict = "❌ REGRESSION DETECTED" if diff.has_regression else "✅ NO REGRESSION"
    print(f"\n  Verdict: {verdict}")
    print(f"{'='*60}\n")



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Model Regression Detection System — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="Run eval suite against a prompt version")
    run_p.add_argument("--version", default=None, help="Prompt version (default: latest)")
    run_p.add_argument("--baseline", default=None, metavar="RUN_ID",
                       help="Baseline run ID to diff against")

   
    sub.add_parser("list-prompts", help="List available prompt versions")

    
    runs_p = sub.add_parser("list-runs", help="Show recent eval runs")
    runs_p.add_argument("--limit", type=int, default=10)

    
    diff_p = sub.add_parser("diff", help="Diff two eval runs")
    diff_p.add_argument("current_run_id")
    diff_p.add_argument("baseline_run_id")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "run":          cmd_run,
        "list-prompts": cmd_list_prompts,
        "list-runs":    cmd_list_runs,
        "diff":         cmd_diff,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()