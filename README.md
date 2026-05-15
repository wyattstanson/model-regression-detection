# 🔍 Model Regression Detection System

A CI/CD pipeline that automatically detects quality regressions whenever you change an LLM prompt or model. Every pull request runs a structured eval suite against a golden dataset, diffs the results against the last accepted baseline, and posts a rich report back to GitHub and Slack — before anything ships.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Running Evals](#running-evals)
6. [Streamlit Dashboard](#streamlit-dashboard)
7. [Docker](#docker)
8. [GitHub Actions CI](#github-actions-ci)
9. [Adding Golden Cases](#adding-golden-cases)
10. [Adding a New Prompt Version](#adding-a-new-prompt-version)
11. [Secrets Reference](#secrets-reference)
12. [Development](#development)

---

## Architecture

```
PR opened / push
       │
       ▼
┌─────────────────────┐
│   GitHub Actions    │  eval.yml workflow
│   (ci_run.py)       │
└────────┬────────────┘
         │  loads
         ▼
┌─────────────────────┐        ┌──────────────────────┐
│   prompt_loader.py  │        │  golden_dataset/      │
│   prompts/vX.Y.Z    │        │  cases.json (60 cases)│
└────────┬────────────┘        └──────────┬───────────┘
         │                               │
         └──────────┬────────────────────┘
                    │ EvalCase[]
                    ▼
         ┌──────────────────┐
         │  eval_runner.py  │  classify_email() × N  →  OpenAI gpt-4o-mini
         │  EvalRun → SQLite│
         └────────┬─────────┘
                  │ current EvalRun
                  ▼
         ┌──────────────────┐
         │  diff_engine.py  │  compare vs baseline EvalRun
         │  DiffResult      │  + slow-drift detection
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  reporter.py     │  HTML report · Slack webhook · PR comment
         └──────────────────┘
```

---

## Project Structure

```
regression-detector/
├── src/
│   ├── models.py          # Pydantic v2: PromptConfig, ClassifierOutput, EvalCase
│   ├── classifier.py      # classify_email(config, text) → ClassifierResult
│   ├── prompt_loader.py   # load_prompt(version), list_prompts()
│   ├── eval_runner.py     # run_eval(config) → EvalRun, SQLite persistence
│   ├── diff_engine.py     # diff_runs(current, baseline) → DiffResult, slow-drift
│   ├── reporter.py        # HTML report, Slack webhook, PR comment
│   └── ci_run.py          # GitHub Actions entrypoint (Click CLI)
├── prompts/
│   ├── v1.0.0.yaml
│   └── v1.1.0.yaml
├── golden_dataset/
│   └── cases.json         # 60 labeled email cases
├── dashboard.py           # Streamlit eval history dashboard
├── tests/
│   └── test_core.py       # 20 unit tests
├── .github/
│   └── workflows/
│       └── eval.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-org/regression-detector.git
cd regression-detector
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY and optionally SLACK_WEBHOOK_URL
```

### 3. Run your first eval

```bash
python -m src.ci_run --prompt-version v1.1.0 --baseline-version v1.0.0
```

An HTML report is written to `reports/` and, if `SLACK_WEBHOOK_URL` is set, a summary is posted to Slack.

---

## Configuration

Create a `.env` file in the project root (never commit this):

```dotenv
# Required
OPENAI_API_KEY=sk-...

# Optional — reporting
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
GITHUB_TOKEN=ghp_...          # for PR comments (set automatically in Actions)
GITHUB_REPOSITORY=org/repo    # set automatically in Actions

# Optional — tuning
DATABASE_URL=./data/evals.db  # SQLite path
REGRESSION_THRESHOLD=0.05     # fail CI if accuracy drops > 5 pp vs baseline
DRIFT_WINDOW=5                # number of runs to include in slow-drift analysis
```

### Prompt YAML schema

```yaml

version: "1.1.0"
model: gpt-4o-mini
temperature: 0.0
max_tokens: 256
system_prompt: |
  You are an email classifier. Classify the email into exactly one category:
  SPAM | HAM | PHISHING | NEWSLETTER
  Respond with JSON: {"label": "<CATEGORY>", "confidence": <0-1>, "reasoning": "..."}
user_template: "Email:\n\n{email_text}"
```

---

## Running Evals

### CLI options

```
python -m src.ci_run [OPTIONS]

Options:
  --prompt-version TEXT    Prompt version to evaluate  [required]
  --baseline-version TEXT  Baseline version to diff against
  --dataset-path PATH      Path to golden dataset JSON  [default: golden_dataset/cases.json]
  --report-dir PATH        Output directory for HTML reports  [default: reports/]
  --fail-on-regression     Exit 1 if regression detected  [default: True]
  --help                   Show this message and exit.
```

### Example

```bash
# Eval v1.1.0 against v1.0.0 baseline, exit non-zero on regression
python -m src.ci_run \
  --prompt-version v1.1.0 \
  --baseline-version v1.0.0 \
  --fail-on-regression
```

### Diff output (stdout)

```
┌─ EvalRun diff: v1.1.0 vs v1.0.0 ──────────────────────────────────────────┐
│  Accuracy:   0.917 → 0.933   Δ +0.017  ✅ improvement
│  Precision:  0.901 → 0.921   Δ +0.020
│  Recall:     0.889 → 0.912   Δ +0.023
│  Latency p50: 1.23 s → 1.18 s
│  Regressions (label flips): 2 cases
│    case_id=14  expected=HAM      got=SPAM   (confidence 0.81)
│    case_id=47  expected=PHISHING got=SPAM   (confidence 0.67)
│  Slow-drift: none detected over last 5 runs
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Streamlit Dashboard

The dashboard shows eval history, per-run metrics, label distributions, and slow-drift charts across all persisted runs.

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501).

**Requirements** (already in `requirements.txt`):
- `streamlit==1.35.0`
- `plotly==5.22.0`
- `pandas==2.2.2`

The dashboard reads directly from the SQLite database at `DATABASE_URL`. Point it at any DB file using the sidebar selector or by setting `DATABASE_URL` in your environment.

---

## Docker

### Build

```bash
docker build -t regression-detector:latest .
```

### Run eval

```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" \
  regression-detector:latest \
  python -m src.ci_run --prompt-version v1.1.0 --baseline-version v1.0.0
```

### Run dashboard

```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -p 8501:8501 \
  regression-detector:latest \
  streamlit run dashboard.py --server.address 0.0.0.0
```

### Run tests

```bash
docker run --rm \
  --env-file .env \
  regression-detector:latest \
  pytest tests/ -v --cov=src --cov-report=term-missing
```

### Multi-stage build summary

| Stage    | Base image        | Purpose                              |
|----------|-------------------|--------------------------------------|
| builder  | python:3.11-slim  | Install deps with pip prefix install |
| runtime  | python:3.11-slim  | Copy /install; run as non-root user  |

The final image contains no build tools, keeping it lean (~350 MB).

---

## GitHub Actions CI

The workflow lives at `.github/workflows/eval.yml` and triggers on every pull request and push to `main`.

### What it does

1. Checks out code
2. Caches pip dependencies
3. Runs `pytest` (unit tests must pass before evals)
4. Runs the eval against the PR's prompt version vs the `main` baseline
5. Posts a PR comment with the diff summary and a link to the uploaded HTML report
6. Posts to Slack (on `main` pushes only)
7. Exits non-zero if accuracy regression > `REGRESSION_THRESHOLD`

### Required GitHub secrets

| Secret               | Where to set           | Description                              |
|----------------------|------------------------|------------------------------------------|
| `OPENAI_API_KEY`     | Repo → Settings → Secrets | Your OpenAI key                       |
| `SLACK_WEBHOOK_URL`  | Repo → Settings → Secrets | Incoming webhook URL (optional)       |

`GITHUB_TOKEN` is provided automatically by Actions — no setup needed.

### Setting secrets via CLI

```bash
gh secret set OPENAI_API_KEY   --body "sk-..."
gh secret set SLACK_WEBHOOK_URL --body "https://hooks.slack.com/services/..."
```

---

## Adding Golden Cases

Edit `golden_dataset/cases.json`. Each case follows this schema:

```jsonc
{
  "id": "case_061",
  "email_text": "Congratulations! You've won a $1000 gift card. Click here to claim.",
  "expected_label": "SPAM",
  "category": "promotional",
  "notes": "Classic prize scam"
}
```

**Tips for a high-quality dataset:**
- Aim for balanced label distribution (SPAM / HAM / PHISHING / NEWSLETTER ≈ 25% each).
- Include edge cases: short emails, non-English text, HTML-heavy bodies, forwarded threads.
- Add a `notes` field explaining why tricky cases have their label — it helps when reviewing regressions.
- After adding cases, run `python -m src.ci_run --prompt-version v1.1.0` locally to verify they pass before pushing.

---

## Adding a New Prompt Version

1. Copy an existing prompt file and bump the version:
   ```bash
   cp prompts/v1.1.0.yaml prompts/v1.2.0.yaml
   ```
2. Edit the system prompt or parameters in `v1.2.0.yaml`.
3. Test locally:
   ```bash
   python -m src.ci_run --prompt-version v1.2.0 --baseline-version v1.1.0
   ```
4. Open a PR — the Actions workflow will run the eval automatically and comment the diff.

---

## Secrets Reference

| Name                  | Required | Used by                       |
|-----------------------|----------|-------------------------------|
| `OPENAI_API_KEY`      | ✅       | `classifier.py` / OpenAI SDK  |
| `DATABASE_URL`        | No       | `eval_runner.py` (default: `./data/evals.db`) |
| `SLACK_WEBHOOK_URL`   | No       | `reporter.py`                 |
| `GITHUB_TOKEN`        | No*      | `reporter.py` PR comment (* auto-injected in Actions) |
| `REGRESSION_THRESHOLD`| No       | `ci_run.py` (default: `0.05`) |
| `DRIFT_WINDOW`        | No       | `diff_engine.py` (default: `5`) |

---

## Development

```bash

ruff check src/ tests/


mypy src/


pytest tests/ -v --cov=src --cov-report=term-missing


OPENAI_API_KEY=sk-... python -m src.ci_run \
  --prompt-version v1.1.0 \
  --baseline-version v1.0.0 \
  --dataset-path golden_dataset/cases.json
```

### Environment variable loading

`python-dotenv` loads `.env` automatically when running via `python -m src.*`. In Docker and Actions, variables are injected directly — no `.env` file is needed.

---

## License

MIT — see `LICENSE`.