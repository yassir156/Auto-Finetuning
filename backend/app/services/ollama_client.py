"""
FineTuneFlow — Ollama Cloud Client Service.

Handles all interactions with the Ollama Cloud API:
  - LLM calls for dataset generation (instruction-tuning & Q&A)
  - JSON repair via LLM fallback
  - Retry & backoff logic
  - JSONL parsing with programmatic repair pipeline
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import (
    OllamaAPIError,
    OllamaAPIKeyMissingError,
)
from app.core.logging import get_logger
from app.services.task_registry import get_task_config, resolve_task_key

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  Prompt Templates (from docs/PROMPTS.md)
# ════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a data generation engine.
You must output ONLY valid JSON Lines (JSONL): one JSON object per line.
No markdown fences, no explanations, no extra text before or after the JSONL.
Each line must be a complete, valid JSON object.
Respect the schema exactly.
If unsure, output fewer examples but ensure every line is valid JSON."""

REPAIR_SYSTEM_PROMPT = """You are a JSON repair tool. You fix broken JSONL text into valid JSONL.
Output ONLY the fixed JSONL lines. No explanations, no markdown."""

REPAIR_USER_TEMPLATE = """Fix the following broken text into valid JSONL.
Each line must be a valid JSON object matching this schema:

SCHEMA: {schema}

RULES:
- Fix JSON syntax errors (missing quotes, brackets, commas).
- If a line is unrecoverable, skip it entirely.
- Do NOT invent new content. Only fix the JSON structure.
- Output ONLY valid JSONL lines.

BROKEN TEXT:
<<<
{bad_text}
>>>

Fixed JSONL:"""


# ════════════════════════════════════════════════════════════
#  HTTP Client with Retry
# ════════════════════════════════════════════════════════════


def _check_api_key() -> None:
    """Ensure an API key is configured."""
    if not settings.OLLAMA_CLOUD_API_KEY or settings.OLLAMA_CLOUD_API_KEY == "CHANGE_ME":
        raise OllamaAPIKeyMissingError()


def _call_ollama_api(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_retries: int = 3,
) -> str:
    """
    Call the Ollama Cloud API with retry and backoff.

    Args:
        messages: Chat messages (system + user).
        model: Model name (defaults to settings.OLLAMA_MODEL).
        temperature: Temperature (defaults to settings.OLLAMA_TEMPERATURE).
        max_tokens: Max tokens (defaults to settings.OLLAMA_MAX_TOKENS).
        max_retries: Number of retries on error.

    Returns:
        The content string from the LLM response.

    Raises:
        OllamaAPIError: On unrecoverable API errors.
    """
    _check_api_key()

    url = f"{settings.OLLAMA_CLOUD_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OLLAMA_CLOUD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or settings.OLLAMA_MODEL,
        "messages": messages,
        "temperature": temperature if temperature is not None else settings.OLLAMA_TEMPERATURE,
        "max_tokens": max_tokens or settings.OLLAMA_MAX_TOKENS,
        "stream": False,
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=settings.OLLAMA_TIMEOUT_SECONDS, follow_redirects=True) as client:
                response = client.post(url, headers=headers, json=payload)

            if response.status_code == 401:
                raise OllamaAPIError(
                    detail="Ollama Cloud API key is invalid",
                    error_code="OLLAMA_API_KEY_INVALID",
                    status_code=401,
                )

            if response.status_code == 404:
                raise OllamaAPIError(
                    detail=f"Ollama model '{payload['model']}' not found",
                    error_code="OLLAMA_MODEL_NOT_FOUND",
                    status_code=404,
                )

            if response.status_code == 429:
                # Rate limited — wait and retry
                wait_time = min(2 ** (attempt + 1), 60)
                last_error = f"Rate limited (429) after {attempt + 1} attempts"
                logger.warning(
                    "ollama.rate_limited",
                    attempt=attempt + 1,
                    wait=wait_time,
                )
                time.sleep(wait_time)
                continue

            if response.status_code >= 500:
                # Server error — retry
                wait_time = min(2 ** attempt, 30)
                logger.warning(
                    "ollama.server_error",
                    status=response.status_code,
                    attempt=attempt + 1,
                    wait=wait_time,
                )
                time.sleep(wait_time)
                last_error = f"Server error {response.status_code}: {response.text[:200]}"
                continue

            response.raise_for_status()

            # Guard against unexpectedly large responses (M6: max 10 MB)
            content_length = int(response.headers.get("content-length", 0))
            if content_length > 10 * 1024 * 1024:
                raise OllamaAPIError(
                    detail=f"Response too large ({content_length} bytes)",
                    error_code="OLLAMA_RESPONSE_TOO_LARGE",
                )

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return content

        except httpx.TimeoutException:
            wait_time = min(2 ** (attempt + 1), 30)
            logger.warning(
                "ollama.timeout",
                attempt=attempt + 1,
                wait=wait_time,
            )
            last_error = f"Timeout after {settings.OLLAMA_TIMEOUT_SECONDS}s"
            time.sleep(wait_time)

        except httpx.HTTPStatusError as e:
            last_error = f"HTTP error: {e.response.status_code}"
            if e.response.status_code < 500:
                raise OllamaAPIError(
                    detail=f"Ollama API error: {e.response.status_code} - {e.response.text[:300]}",
                    error_code="OLLAMA_API_ERROR",
                )
            # Server errors — retry
            wait_time = min(2 ** attempt, 30)
            time.sleep(wait_time)

        except OllamaAPIError:
            raise

        except Exception as e:
            last_error = str(e)
            wait_time = min(2 ** attempt, 30)
            logger.warning(
                "ollama.unexpected_error",
                error=str(e),
                attempt=attempt + 1,
                wait=wait_time,
            )
            time.sleep(wait_time)

    raise OllamaAPIError(
        detail=f"Ollama API failed after {max_retries} attempts: {last_error}",
        error_code="OLLAMA_API_ERROR",
    )


# ════════════════════════════════════════════════════════════
#  JSONL Parsing & Repair Pipeline
# ════════════════════════════════════════════════════════════


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```)."""
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"```(?:json|jsonl)?\s*\n?", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _strip_llm_prefixes(text: str) -> str:
    """Remove common LLM preamble text before the actual JSONL."""
    # Common prefixes LLMs add
    prefixes = [
        r"^Here (?:are|is) (?:the )?\d* ?(?:training )?examples?:?\s*\n?",
        r"^(?:Sure|Okay|Of course)[!.,]\s*(?:Here (?:are|is))?\s*\n?",
        r"^Output:?\s*\n?",
        r"^JSONL:?\s*\n?",
    ]
    for pattern in prefixes:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()


def _fix_trailing_commas(text: str) -> str:
    """Fix trailing commas in JSON objects."""
    return re.sub(r",\s*}", "}", text)


def _fix_common_json_issues(text: str) -> str:
    """Fix common JSON issues in a single line."""
    text = _fix_trailing_commas(text)
    # Fix single quotes to double quotes (simple heuristic)
    # Only if no double quotes present
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    return text


def _extract_json_objects(text: str) -> list[str]:
    """Extract JSON objects from text using regex."""
    # Match JSON objects (non-nested, which is our case)
    objects = re.findall(r"\{[^{}]*\}", text)
    return objects


def parse_jsonl_response(raw_content: str, task_type: str = "instruction_tuning") -> tuple[list[dict], list[str]]:
    """
    Parse an LLM response expected to contain JSONL lines.

    Uses the task_registry to determine which fields are required.

    Returns:
        (valid_examples, failed_lines) — parsed dicts and un-parseable lines.
    """
    config = get_task_config(task_type)
    required = set(config.required_fields)

    def _is_valid_obj(obj: dict) -> bool:
        """Check if obj has all required fields."""
        if not isinstance(obj, dict):
            return False
        return all(f in obj for f in required)

    def _fill_defaults(obj: dict) -> dict:
        """Fill optional fields with defaults."""
        for f in config.optional_fields:
            if f not in obj:
                obj[f] = ""
        return obj

    # Step 1-2: Clean up
    cleaned = _strip_markdown_fences(raw_content)
    cleaned = _strip_llm_prefixes(cleaned)

    valid: list[dict] = []
    failed: list[str] = []

    lines = cleaned.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Step 3: Try direct parse
        try:
            obj = json.loads(line)
            if _is_valid_obj(obj):
                valid.append(_fill_defaults(obj))
                continue
        except json.JSONDecodeError:
            pass

        # Step 4: Fix common issues and retry
        fixed = _fix_common_json_issues(line)
        try:
            obj = json.loads(fixed)
            if _is_valid_obj(obj):
                valid.append(_fill_defaults(obj))
                continue
        except json.JSONDecodeError:
            pass

        # Step 5: Try to extract JSON objects via regex
        extracted = _extract_json_objects(line)
        found = False
        for fragment in extracted:
            try:
                obj = json.loads(fragment)
                if _is_valid_obj(obj):
                    valid.append(_fill_defaults(obj))
                    found = True
                    break
            except json.JSONDecodeError:
                continue

        if not found:
            failed.append(line)

    return valid, failed


def _repair_via_llm(bad_text: str, task_type: str = "instruction_tuning", model: str | None = None) -> list[dict]:
    """
    Attempt to repair broken JSONL by calling the LLM.

    This is a last-resort fallback — called once per generation.
    """
    logger.info("ollama.repair.start", bad_lines=bad_text.count("\n") + 1)

    config = get_task_config(task_type)

    messages = [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": REPAIR_USER_TEMPLATE.format(
            schema=config.repair_schema, bad_text=bad_text
        )},
    ]

    try:
        content = _call_ollama_api(
            messages=messages,
            model=model,
            temperature=0.1,  # Very deterministic for repair
            max_retries=2,
        )
        repaired, still_failed = parse_jsonl_response(content, task_type=task_type)
        logger.info(
            "ollama.repair.done",
            repaired=len(repaired),
            still_failed=len(still_failed),
        )
        return repaired
    except Exception as e:
        logger.warning("ollama.repair.failed", error=str(e))
        return []


# ════════════════════════════════════════════════════════════
#  Main Public API
# ════════════════════════════════════════════════════════════


def generate_examples_from_chunk(
    chunk_text: str,
    task_type: str = "instruction_tuning",
    num_examples: int = 5,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Generate dataset examples from a text chunk using Ollama Cloud.

    Args:
        chunk_text: The source text chunk.
        task_type: One of the 7 supported task types.
        num_examples: Number of examples to generate.
        model: Optional model override.

    Returns:
        List of validated example dicts.
    """
    config = get_task_config(task_type)
    template = config.prompt_template

    user_prompt = template.format(N=num_examples, chunk_text=chunk_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    logger.info(
        "ollama.generate.start",
        task_type=task_type,
        num_examples=num_examples,
        chunk_length=len(chunk_text),
    )

    # ── Call Ollama API ──
    raw_content = _call_ollama_api(messages=messages, model=model)

    # ── Parse response ──
    valid_examples, failed_lines = parse_jsonl_response(raw_content, task_type=task_type)

    logger.info(
        "ollama.generate.parsed",
        valid=len(valid_examples),
        failed=len(failed_lines),
    )

    # ── LLM repair for failures (if any) ──
    if failed_lines and len(failed_lines) > 0:
        bad_text = "\n".join(failed_lines)
        repaired = _repair_via_llm(bad_text, task_type=task_type, model=model)
        valid_examples.extend(repaired)
        logger.info(
            "ollama.generate.after_repair",
            total_valid=len(valid_examples),
        )

    return valid_examples
