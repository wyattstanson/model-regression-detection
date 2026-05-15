
"""
Model Regression Detection System — source package.

Exports the public surface used by ci_run.py and dashboard.py.
"""

from .models import PromptConfig, ClassifierOutput, EvalCase, ClassifierResult
from .prompt_loader import load_prompt, list_prompts
from .classifier import classify_email
from .eval_runner import run_eval
from .diff_engine import diff_runs

__all__ = [
    "PromptConfig",
    "ClassifierOutput",
    "EvalCase",
    "ClassifierResult",
    "load_prompt",
    "list_prompts",
    "classify_email",
    "run_eval",
    "diff_runs",
]