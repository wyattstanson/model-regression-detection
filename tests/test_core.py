
"""
20 unit tests covering models, prompt loader, classifier, eval runner, and diff engine.

Run with:
  pytest tests/test_core.py -v

Dependencies (all in requirements-dev.txt):
  pytest, pytest-mock, pydantic
"""

from __future__ import annotations

import json
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    CaseDiff,
    ClassifierOutput,
    ClassifierResult,
    DiffResult,
    EmailLabel,
    EvalCase,
    EvalRun,
    PromptConfig,
    RunStatus,
)
from src.prompt_loader import invalidate_cache, list_prompts, load_prompt
from src.diff_engine import diff_runs




@pytest.fixture()
def minimal_config() -> PromptConfig:
    return PromptConfig(
        version="1.0.0",
        name="Test Config",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=256,
        system_prompt="You are a classifier.",
        user_prompt_template="Classify: {email_text}",
    )


@pytest.fixture()
def sample_result(minimal_config) -> ClassifierResult:
    output = ClassifierOutput(label=EmailLabel.SPAM, confidence=0.95, reasoning="Clearly spam.")
    return ClassifierResult(
        case_id="case-001",
        output=output,
        prompt_version=minimal_config.version,
        model=minimal_config.model,
        latency_ms=123.4,
        input_tokens=50,
        output_tokens=20,
    )


@pytest.fixture()
def sample_case() -> EvalCase:
    return EvalCase(
        id="case-001",
        email_text="Win a free iPhone now!",
        expected_label=EmailLabel.SPAM,
    )


@pytest.fixture()
def tmp_prompts_dir(tmp_path: Path, minimal_config: PromptConfig) -> Path:
    """Temp directory with a single v1.0.0.yaml prompt file."""
    import yaml
    d = tmp_path / "prompts"
    d.mkdir()
    data = {
        "version": "1.0.0",
        "name": "Test",
        "description": "",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 256,
        "system_prompt": "You classify emails.",
        "user_prompt_template": "Classify: {email_text}",
    }
    (d / "v1.0.0.yaml").write_text(yaml.dump(data))
    invalidate_cache()
    return d




class TestPromptConfig:
    def test_valid_config_creates_successfully(self, minimal_config):
        assert minimal_config.version == "1.0.0"

    def test_invalid_semver_raises(self):
        with pytest.raises(ValueError, match="semver"):
            PromptConfig(
                version="v1.0",
                name="X",
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=256,
                system_prompt="sys",
                user_prompt_template="{email_text}",
            )

    def test_missing_placeholder_raises(self):
        with pytest.raises(ValueError, match="email_text"):
            PromptConfig(
                version="1.0.0",
                name="X",
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=256,
                system_prompt="sys",
                user_prompt_template="No placeholder here",
            )

    def test_render_user_prompt_substitutes_text(self, minimal_config):
        rendered = minimal_config.render_user_prompt("Hello World")
        assert "Hello World" in rendered




class TestClassifierOutput:
    def test_valid_output(self):
        o = ClassifierOutput(label=EmailLabel.INQUIRY, confidence=0.88, reasoning="Question email.")
        assert o.label == EmailLabel.INQUIRY

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ClassifierOutput(label=EmailLabel.SPAM, confidence=1.5, reasoning="Too confident.")

    def test_confidence_is_rounded(self):
        o = ClassifierOutput(label=EmailLabel.SPAM, confidence=0.987654321, reasoning="x")
        assert len(str(o.confidence).split(".")[-1]) <= 4




class TestEvalCase:
    def test_eval_case_strips_whitespace(self):
        case = EvalCase(email_text="  hello  ", expected_label=EmailLabel.SPAM)
        assert case.email_text == "hello"

    def test_eval_case_auto_id(self):
        case = EvalCase(email_text="test", expected_label=EmailLabel.OTHER)
        assert case.id  # auto-generated

    def test_eval_case_empty_text_raises(self):
        with pytest.raises(ValueError):
            EvalCase(email_text="", expected_label=EmailLabel.SPAM)




class TestClassifierResult:
    def test_total_tokens(self, sample_result):
        assert sample_result.total_tokens == 70

    def test_error_defaults_to_none(self, sample_result):
        assert sample_result.error is None




class TestPromptLoader:
    def test_list_prompts_returns_sorted_versions(self, tmp_prompts_dir, tmp_path):
        import yaml
        # Add a second version
        data2 = {
            "version": "1.1.0",
            "name": "v2",
            "description": "",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 256,
            "system_prompt": "sys",
            "user_prompt_template": "{email_text}",
        }
        (tmp_prompts_dir / "v1.1.0.yaml").write_text(yaml.dump(data2))
        invalidate_cache()
        versions = list_prompts(tmp_prompts_dir)
        assert versions == ["1.0.0", "1.1.0"]

    def test_load_prompt_by_version(self, tmp_prompts_dir):
        config = load_prompt("1.0.0", prompts_dir=tmp_prompts_dir)
        assert config.version == "1.0.0"

    def test_load_prompt_latest_alias(self, tmp_prompts_dir):
        config = load_prompt("latest", prompts_dir=tmp_prompts_dir)
        assert config.version == "1.0.0"

    def test_load_nonexistent_version_raises(self, tmp_prompts_dir):
        with pytest.raises(FileNotFoundError):
            load_prompt("9.9.9", prompts_dir=tmp_prompts_dir)




class TestEvalRunMetrics:
    def test_compute_metrics_avg_latency(self, sample_result, minimal_config):
        r2 = sample_result.model_copy(update={"case_id": "case-002", "latency_ms": 200.0})
        run = EvalRun(prompt_version="1.0.0", model="gpt-4o-mini", results=[sample_result, r2])
        run.compute_metrics()
        assert run.avg_latency_ms == pytest.approx((123.4 + 200.0) / 2, rel=1e-3)




def _make_result(case_id, label, confidence=0.9, latency=100.0, error=None):
    output = ClassifierOutput(label=label, confidence=confidence, reasoning="x")
    return ClassifierResult(
        case_id=case_id,
        output=output,
        prompt_version="1.0.0",
        model="gpt-4o-mini",
        latency_ms=latency,
        error=error,
    )


def _make_run(version, results, accuracy=None):
    run = EvalRun(prompt_version=version, model="gpt-4o-mini", results=results)
    run.accuracy = accuracy
    run.avg_latency_ms = sum(r.latency_ms for r in results) / len(results) if results else 0
    run.avg_confidence = 0.9
    return run


class TestDiffEngine:
    def test_regression_detected(self):
        baseline_results = [_make_result("c1", EmailLabel.SPAM)]
        current_results  = [_make_result("c1", EmailLabel.OTHER)]  # was correct, now wrong
        cases            = [EvalCase(id="c1", email_text="spam!", expected_label=EmailLabel.SPAM)]

        baseline = _make_run("1.0.0", baseline_results, accuracy=1.0)
        current  = _make_run("1.1.0", current_results,  accuracy=0.0)

        diff = diff_runs(current, baseline, cases)
        assert len(diff.regressions) == 1
        assert diff.has_regression is True

    def test_improvement_detected(self):
        baseline_results = [_make_result("c1", EmailLabel.OTHER)]   # wrong
        current_results  = [_make_result("c1", EmailLabel.SPAM)]    # now correct
        cases            = [EvalCase(id="c1", email_text="spam!", expected_label=EmailLabel.SPAM)]

        baseline = _make_run("1.0.0", baseline_results, accuracy=0.0)
        current  = _make_run("1.1.0", current_results,  accuracy=1.0)

        diff = diff_runs(current, baseline, cases)
        assert len(diff.improvements) == 1
        assert not diff.has_regression

    def test_slow_drift_flagged(self):
        baseline_results = [_make_result("c1", EmailLabel.SPAM, latency=100.0)]
        current_results  = [_make_result("c1", EmailLabel.SPAM, latency=700.0)]  # +600ms
        cases            = [EvalCase(id="c1", email_text="spam!", expected_label=EmailLabel.SPAM)]

        baseline = _make_run("1.0.0", baseline_results)
        current  = _make_run("1.1.0", current_results)

        diff = diff_runs(current, baseline, cases)
        assert "c1" in diff.slow_drift_cases

    def test_no_change_no_regression(self):
        results = [_make_result("c1", EmailLabel.SPAM)]
        cases   = [EvalCase(id="c1", email_text="spam!", expected_label=EmailLabel.SPAM)]

        baseline = _make_run("1.0.0", results, accuracy=1.0)
        current  = _make_run("1.0.0", results, accuracy=1.0)

        diff = diff_runs(current, baseline, cases)
        assert not diff.has_regression
        assert diff.regression_rate == 0.0