

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import openai
from openai import OpenAI

from .models import (
    ClassifierOutput,
    ClassifierResult,
    EmailLabel,
    PromptConfig,
)

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()          # reads OPENAI_API_KEY from env
    return _client



MAX_RETRIES = 3
BASE_BACKOFF_S = 1.0          # seconds; doubled each retry
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}

_FALLBACK_OUTPUT = ClassifierOutput(
    label=EmailLabel.OTHER,
    confidence=0.0,
    reasoning="Classification failed; fallback applied.",
)




def classify_email(
    config: PromptConfig,
    email_text: str,
    case_id: str = "unknown",
) -> ClassifierResult:
    """
    Classify a single email using the provided PromptConfig.

    Parameters
    ----------
    config     : PromptConfig  — versioned prompt + model params
    email_text : str           — raw email body
    case_id    : str           — identifier for logging / tracing

    Returns
    -------
    ClassifierResult  — always returns; errors stored in .error field
    """
    user_prompt = config.render_user_prompt(email_text)
    start_ms = time.monotonic() * 1000

    output, error, in_tok, out_tok = _call_with_retry(
        config=config,
        user_prompt=user_prompt,
        case_id=case_id,
    )

    latency_ms = time.monotonic() * 1000 - start_ms

    return ClassifierResult(
        case_id=case_id,
        output=output,
        prompt_version=config.version,
        model=config.model,
        latency_ms=round(latency_ms, 2),
        input_tokens=in_tok,
        output_tokens=out_tok,
        error=error,
    )




def _call_with_retry(
    config: PromptConfig,
    user_prompt: str,
    case_id: str,
) -> tuple[ClassifierOutput, Optional[str], int, int]:
    """
    Returns (output, error_message | None, input_tokens, output_tokens).
    Retries on transient errors; returns fallback output on permanent failure.
    """
    last_error: str = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _get_client().chat.completions.create(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )

            raw_text   = response.choices[0].message.content or ""
            in_tokens  = response.usage.prompt_tokens     if response.usage else 0
            out_tokens = response.usage.completion_tokens if response.usage else 0

            output = _parse_output(raw_text, case_id)
            return output, None, in_tokens, out_tokens

        except openai.RateLimitError as exc:
            last_error = f"RateLimitError: {exc}"
            logger.warning("[%s] attempt %d/%d — rate limited, backing off", case_id, attempt, MAX_RETRIES)
        except openai.APIStatusError as exc:
            last_error = f"APIStatusError({exc.status_code}): {exc.message}"
            if exc.status_code not in RETRYABLE_HTTP_CODES:
                logger.error("[%s] non-retryable API error: %s", case_id, last_error)
                break
            logger.warning("[%s] attempt %d/%d — %s", case_id, attempt, MAX_RETRIES, last_error)
        except openai.APIConnectionError as exc:
            last_error = f"APIConnectionError: {exc}"
            logger.warning("[%s] attempt %d/%d — connection error", case_id, attempt, MAX_RETRIES)
        except _ParseError as exc:
            last_error = str(exc)
            logger.warning("[%s] parse failure (no retry): %s", case_id, last_error)
            break   
        except Exception as exc:  
            last_error = f"Unexpected: {exc}"
            logger.exception("[%s] unexpected error on attempt %d", case_id, attempt)
            break

        if attempt < MAX_RETRIES:
            backoff = BASE_BACKOFF_S * (2 ** (attempt - 1))
            time.sleep(backoff)

    logger.error("[%s] all attempts exhausted. last_error=%s", case_id, last_error)
    return _FALLBACK_OUTPUT, last_error, 0, 0


class _ParseError(Exception):
    pass


def _parse_output(raw: str, case_id: str) -> ClassifierOutput:
    """Parse and validate the model's JSON response into ClassifierOutput."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _ParseError(f"[{case_id}] JSON decode failed: {exc}  raw={raw!r}") from exc

    try:
        return ClassifierOutput.model_validate(data)
    except Exception as exc:
        raise _ParseError(f"[{case_id}] Pydantic validation failed: {exc}  data={data}") from exc