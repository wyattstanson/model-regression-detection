"""
eval_runner.py — Core evaluation engine.

Runs every case in the golden dataset through the classifier,
scores on multiple dimensions, stores results in SQLite,
and returns a structured EvalRun ready for diffing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import openai

from src.classifier import classify_email, ClassifierResult
from src.models import EvalCase, PromptConfig, EmailCategory

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DATABASE_URL", str(Path(__file__).parent.parent / "data" / "demo.db")))
GOLDEN_DATASET_PATH = Path(__file__).parent.parent / "golden_dataset" / "cases.json"




@dataclass
class CaseScore:
    case_id: str
    passed: bool                      
    category_match: bool
    summary_relevance: int            
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None
    predicted_category: Optional[str] = None
    predicted_summary: Optional[str] = None




@dataclass
class EvalRun:
    run_id: str
    prompt_version: str
    model: str
    timestamp: str
    total_cases: int
    passed: int
    failed: int
    pass_rate: float
    avg_latency_ms: float
    total_tokens: int
    per_category_accuracy: dict[str, float]
    case_scores: list[CaseScore] = field(default_factory=list)

    @property
    def failed_cases(self) -> list[CaseScore]:
        return [c for c in self.case_scores if not c.passed]

    @property
    def passed_cases(self) -> list[CaseScore]:
        return [c for c in self.case_scores if c.passed]




JUDGE_SYSTEM = """You are an evaluation judge for a customer support classifier.
Rate how well a generated summary captures the core issue, given the expected keywords.
Return ONLY a JSON object: {"score": <int 1-5>, "reason": "<one sentence>"}
1=completely off, 3=captures main point, 5=excellent and specific."""


def score_summary(
    email_text: str,
    summary: str,
    keywords: list[str],
    client: openai.OpenAI,
    model: str = "gpt-4o-mini",
) -> int:
    """Returns 1–5 relevance score. Returns 0 on failure (don't penalise)."""
    if not keywords:
        return 3  

    prompt = (
        f"Email: {email_text[:300]}\n"
        f"Expected keywords: {', '.join(keywords)}\n"
        f"Generated summary: {summary}\n\n"
        "Rate the summary quality 1–5."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=80,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return max(1, min(5, int(data.get("score", 3))))
    except Exception as e:
        logger.warning(f"Judge failed for case: {e}")
        return 0




def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[EvalCase]:
    with path.open() as f:
        raw = json.load(f)
    return [EvalCase(**item) for item in raw]




def run_eval(
    config: PromptConfig,
    use_judge: bool = True,
    dataset_path: Path = GOLDEN_DATASET_PATH,
    client: Optional[openai.OpenAI] = None,
) -> EvalRun:
    """
    Run the full evaluation pipeline.

    Args:
        config:       The PromptConfig version under test.
        use_judge:    Whether to call LLM-as-judge for summary scoring (costs tokens).
        dataset_path: Path to the golden dataset JSON.
        client:       Optional pre-built OpenAI client.

    Returns:
        EvalRun with all scores and metadata.
    """
    client = client or openai.OpenAI()
    cases = load_golden_dataset(dataset_path)
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(f"Starting eval run {run_id} | prompt={config.version} | cases={len(cases)}")

    case_scores: list[CaseScore] = []

    for i, case in enumerate(cases, 1):
        logger.info(f"  [{i}/{len(cases)}] {case.id}")

        result: ClassifierResult = classify_email(config, case.email_text, client=client)

        if not result.success:
            case_scores.append(CaseScore(
                case_id=case.id,
                passed=False,
                category_match=False,
                summary_relevance=0,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                error=result.error,
            ))
            continue

        out = result.output
        category_match = out.category == case.expected_category

        relevance = 0
        if use_judge and out.summary:
            relevance = score_summary(
                case.email_text, out.summary,
                case.expected_summary_keywords, client, config.model
            )

        case_scores.append(CaseScore(
            case_id=case.id,
            passed=category_match,
            category_match=category_match,
            summary_relevance=relevance,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            predicted_category=out.category.value,
            predicted_summary=out.summary,
        ))


        time.sleep(0.1)

    

    total = len(case_scores)
    passed = sum(1 for s in case_scores if s.passed)
    pass_rate = passed / total if total > 0 else 0.0

    avg_latency = (
        sum(s.latency_ms for s in case_scores) / total if total > 0 else 0.0
    )
    total_tokens = sum(s.prompt_tokens + s.completion_tokens for s in case_scores)

    
    categories = [c.value for c in EmailCategory]
    per_cat: dict[str, float] = {}
    for cat in categories:
        cat_cases = [
            s for s, c in zip(case_scores, cases)
            if c.expected_category.value == cat
        ]
        if cat_cases:
            per_cat[cat] = sum(1 for s in cat_cases if s.passed) / len(cat_cases)
        else:
            per_cat[cat] = 0.0

    run = EvalRun(
        run_id=run_id,
        prompt_version=config.version,
        model=config.model,
        timestamp=timestamp,
        total_cases=total,
        passed=passed,
        failed=total - passed,
        pass_rate=round(pass_rate, 4),
        avg_latency_ms=round(avg_latency, 2),
        total_tokens=total_tokens,
        per_category_accuracy=per_cat,
        case_scores=case_scores,
    )

    _save_run(run)
    logger.info(
        f"Eval complete: {passed}/{total} passed ({pass_rate:.1%}) | "
        f"latency={avg_latency:.0f}ms | tokens={total_tokens}"
    )
    return run



def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS eval_runs (
        run_id          TEXT PRIMARY KEY,
        prompt_version  TEXT NOT NULL,
        model           TEXT NOT NULL,
        timestamp       TEXT NOT NULL,
        total_cases     INTEGER,
        passed          INTEGER,
        failed          INTEGER,
        pass_rate       REAL,
        avg_latency_ms  REAL,
        total_tokens    INTEGER,
        per_category_accuracy TEXT,
        case_scores_json      TEXT
    );
    """)
    conn.commit()


def _save_run(run: EvalRun) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO eval_runs VALUES
           (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run.run_id,
            run.prompt_version,
            run.model,
            run.timestamp,
            run.total_cases,
            run.passed,
            run.failed,
            run.pass_rate,
            run.avg_latency_ms,
            run.total_tokens,
            json.dumps(run.per_category_accuracy),
            json.dumps([asdict(s) for s in run.case_scores]),
        ),
    )
    conn.commit()
    conn.close()
    logger.info(f"Run {run.run_id} saved to {DB_PATH}")


def load_recent_runs(n: int = 10) -> list[dict]:
    """Load the last N runs from the database (without case_scores for speed)."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT run_id, prompt_version, model, timestamp,
                  total_cases, passed, failed, pass_rate,
                  avg_latency_ms, total_tokens, per_category_accuracy
           FROM eval_runs ORDER BY timestamp DESC LIMIT ?""",
        (n,),
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["per_category_accuracy"] = json.loads(d["per_category_accuracy"])
        result.append(d)
    return result


def load_run_by_id(run_id: str) -> Optional[EvalRun]:
    """Load a full run including case scores."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    scores = [CaseScore(**s) for s in json.loads(d.pop("case_scores_json"))]
    d["per_category_accuracy"] = json.loads(d["per_category_accuracy"])
    return EvalRun(**d, case_scores=scores)