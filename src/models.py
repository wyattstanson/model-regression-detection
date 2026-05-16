
"""
Core Pydantic v2 data models for the regression detection system.

Hierarchy:
  PromptConfig       — versioned prompt + model params loaded from YAML
  ClassifierOutput   — raw JSON response from the LLM
  ClassifierResult   — ClassifierOutput enriched with metadata (case_id, latency, etc.)
  EvalCase           — one row in the golden dataset
  EvalRun            — result of running an entire eval suite
  DiffResult         — comparison between two EvalRuns
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class EmailCategory(str, Enum):
    SPAM = "SPAM"
    HAM = "HAM"
    PHISHING = "PHISHING"
    NEWSLETTER = "NEWSLETTER"

class EmailLabel(str, Enum):
    SPAM      = "SPAM"
    PHISHING  = "PHISHING"
    COMPLAINT = "COMPLAINT"
    INQUIRY   = "INQUIRY"
    SUPPORT   = "SUPPORT"
    FEEDBACK  = "FEEDBACK"
    OTHER     = "OTHER"



class RunStatus(str, Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"



class PromptConfig(BaseModel):
    """Versioned prompt configuration loaded from a YAML file."""

    version: str = Field(..., description="Semver string, e.g. '1.0.0'")
    name: str
    description: str = ""
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, gt=0)
    system_prompt: str
    user_prompt_template: str = Field(
        ...,
        description="Must contain {email_text} placeholder.",
    )

    @field_validator("version")
    @classmethod
    def version_is_semver(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"version must be semver (X.Y.Z), got '{v}'")
        return v

    @field_validator("user_prompt_template")
    @classmethod
    def template_has_placeholder(cls, v: str) -> str:
        if "{email_text}" not in v:
            raise ValueError("user_prompt_template must contain {email_text}")
        return v

    def render_user_prompt(self, email_text: str) -> str:
        return self.user_prompt_template.format(email_text=email_text)



class ClassifierOutput(BaseModel):
    """Raw structured output from the LLM (parsed from JSON response)."""

    label: EmailLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., max_length=200)

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)


class ClassifierResult(BaseModel):
    """ClassifierOutput enriched with request metadata."""

    case_id: str
    output: ClassifierOutput
    prompt_version: str
    model: str
    latency_ms: float = Field(..., ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens




class EvalCase(BaseModel):
    """One labelled example in the golden dataset."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    email_text: str = Field(..., min_length=1)
    expected_label: EmailLabel
    source: str = Field(default="manual", description="Origin of this test case.")
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("email_text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()



class EvalRun(BaseModel):
    """Aggregated result of running the full eval suite against one prompt version."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt_version: str
    model: str
    status: RunStatus = RunStatus.PENDING
    results: list[ClassifierResult] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None

  
    accuracy: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    avg_confidence: Optional[float] = None
    total_cost_usd: Optional[float] = None

    _INPUT_COST_PER_M:  float = 0.150
    _OUTPUT_COST_PER_M: float = 0.600

    def compute_metrics(self) -> None:
        """Populate aggregate fields from self.results. Call after all results are in."""
        completed = [r for r in self.results if r.error is None]
        if not completed:
            return

        correct = sum(
            1 for r in completed
            
        )
        self.avg_latency_ms = round(
            sum(r.latency_ms for r in completed) / len(completed), 2
        )
        self.avg_confidence = round(
            sum(r.output.confidence for r in completed) / len(completed), 4
        )
        total_input  = sum(r.input_tokens  for r in completed)
        total_output = sum(r.output_tokens for r in completed)
        self.total_cost_usd = round(
            (total_input  / 1_000_000) * self._INPUT_COST_PER_M
            + (total_output / 1_000_000) * self._OUTPUT_COST_PER_M,
            6,
        )
        self.finished_at = datetime.utcnow()
        self.status = RunStatus.COMPLETED


class CaseDiff(BaseModel):
    """Per-case comparison between two runs."""

    case_id: str
    baseline_label: Optional[EmailLabel]
    current_label: Optional[EmailLabel]
    baseline_confidence: Optional[float]
    current_confidence: Optional[float]
    label_changed: bool
    confidence_delta: Optional[float]  


class DiffResult(BaseModel):
    """Comparison between a current EvalRun and a baseline EvalRun."""

    baseline_run_id: str
    current_run_id: str
    baseline_version: str
    current_version: str

    accuracy_delta: Optional[float]       
    latency_delta_ms: Optional[float]
    confidence_delta: Optional[float]

    regressions: list[CaseDiff] = Field(default_factory=list)  
    improvements: list[CaseDiff] = Field(default_factory=list)  
    neutral_changes: list[CaseDiff] = Field(default_factory=list)

    slow_drift_cases: list[str] = Field(
        default_factory=list,
        description="case_ids where latency increased by > threshold.",
    )

    @property
    def has_regression(self) -> bool:
        return len(self.regressions) > 0

    @property
    def regression_rate(self) -> float:
        total = len(self.regressions) + len(self.improvements) + len(self.neutral_changes)
        return len(self.regressions) / total if total else 0.0