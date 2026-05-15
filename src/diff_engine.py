
"""
diff_runs — compare a current EvalRun against a baseline and surface regressions.

Outputs
-------
DiffResult with:
  - regressions    : cases that were correct in baseline but wrong in current
  - improvements   : cases that were wrong in baseline but correct in current
  - neutral_changes: label changed but correctness status unchanged
  - slow_drift     : case_ids where latency increased by > LATENCY_DRIFT_THRESHOLD_MS
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import (
    CaseDiff,
    ClassifierResult,
    DiffResult,
    EvalCase,
    EvalRun,
    EmailLabel,
)

logger = logging.getLogger(__name__)


LATENCY_DRIFT_THRESHOLD_MS = 500.0   # flag if current latency > baseline + this
CONFIDENCE_ALERT_DELTA     = -0.10   # flag if confidence drops more than 10 pp


def diff_runs(
    current:  EvalRun,
    baseline: EvalRun,
    cases:    Optional[list[EvalCase]] = None,
) -> DiffResult:
    """
    Compare *current* against *baseline* and return a DiffResult.

    Parameters
    ----------
    current  : the newly evaluated run
    baseline : the reference run (e.g. from main branch / last release)
    cases    : optional EvalCase list to resolve expected_label; if omitted,
               correctness classification is skipped (only label-change analysis)
    """
    baseline_map = {r.case_id: r for r in baseline.results}
    current_map  = {r.case_id: r for r in current.results}
    case_map     = {c.id: c for c in cases} if cases else {}

    all_ids = set(baseline_map) | set(current_map)

    regressions:     list[CaseDiff] = []
    improvements:    list[CaseDiff] = []
    neutral_changes: list[CaseDiff] = []
    slow_drift:      list[str]      = []

    for cid in sorted(all_ids):
        b_res = baseline_map.get(cid)
        c_res = current_map.get(cid)

        b_label  = b_res.output.label      if b_res else None
        c_label  = c_res.output.label      if c_res else None
        b_conf   = b_res.output.confidence if b_res else None
        c_conf   = c_res.output.confidence if c_res else None
        b_lat    = b_res.latency_ms        if b_res else None
        c_lat    = c_res.latency_ms        if c_res else None

        label_changed    = (b_label != c_label)
        confidence_delta = round(c_conf - b_conf, 4) if (c_conf is not None and b_conf is not None) else None

        diff = CaseDiff(
            case_id=cid,
            baseline_label=b_label,
            current_label=c_label,
            baseline_confidence=b_conf,
            current_confidence=c_conf,
            label_changed=label_changed,
            confidence_delta=confidence_delta,
        )

      
        if cid in case_map:
            expected     = case_map[cid].expected_label
            b_correct    = (b_label == expected) if b_label is not None else False
            c_correct    = (c_label == expected) if c_label is not None else False

            if b_correct and not c_correct:
                regressions.append(diff)
            elif not b_correct and c_correct:
                improvements.append(diff)
            elif label_changed:
                neutral_changes.append(diff)
        elif label_changed:
            neutral_changes.append(diff)

       
        if b_lat is not None and c_lat is not None:
            if c_lat - b_lat > LATENCY_DRIFT_THRESHOLD_MS:
                slow_drift.append(cid)
                logger.debug(
                    "Slow drift detected for case %s: baseline=%.0fms current=%.0fms",
                    cid, b_lat, c_lat,
                )

   
    acc_delta = (
        round((current.accuracy or 0) - (baseline.accuracy or 0), 4)
        if current.accuracy is not None and baseline.accuracy is not None
        else None
    )
    lat_delta = (
        round((current.avg_latency_ms or 0) - (baseline.avg_latency_ms or 0), 2)
        if current.avg_latency_ms is not None and baseline.avg_latency_ms is not None
        else None
    )
    conf_delta = (
        round((current.avg_confidence or 0) - (baseline.avg_confidence or 0), 4)
        if current.avg_confidence is not None and baseline.avg_confidence is not None
        else None
    )

    result = DiffResult(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        baseline_version=baseline.prompt_version,
        current_version=current.prompt_version,
        accuracy_delta=acc_delta,
        latency_delta_ms=lat_delta,
        confidence_delta=conf_delta,
        regressions=regressions,
        improvements=improvements,
        neutral_changes=neutral_changes,
        slow_drift_cases=slow_drift,
    )

    logger.info(
        "Diff %s→%s: regressions=%d improvements=%d neutral=%d slow_drift=%d",
        baseline.prompt_version, current.prompt_version,
        len(regressions), len(improvements), len(neutral_changes), len(slow_drift),
    )
    return result