"""Shared, resilient OpenRouter helpers (retry/backoff + robust JSON extraction).

Free OpenRouter endpoints frequently answer 503 (provider unavailable) or 429
(per-minute rate limit), and many free models are "reasoning" models that spend
the token budget thinking before emitting any answer. This module handles both.
"""
from __future__ import annotations

import json
import random
import re
import time
import os

import openai

# Enough room for a reasoning model to think AND still emit the ~100-token JSON.
DEFAULT_MAX_TOKENS = 2000
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_EVALUATION_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
)


def provider_settings():
    """Return provider, key, base URL, and default model without printing secrets."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("MODEL_NAME", "gemini-3.8-flash")
        if model.startswith("openrouter/") or model.endswith(":free"):
            model = "gemini-3.8-flash"
        return provider, key, GEMINI_BASE_URL, model
    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("MODEL_NAME", "openrouter/free")
        return provider, key, OPENROUTER_BASE_URL, model
    raise ValueError("LLM_PROVIDER must be 'gemini' or 'openrouter'")

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class EmptyResponseError(RuntimeError):
    """Provider returned HTTP 200 but no usable choices (OpenRouter does this on upstream errors)."""


def _retry_after(exc):
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _is_retryable(exc):
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, EmptyResponseError)):
        return True
    # Daily free-tier exhaustion cannot be fixed by retrying; avoid wasting
    # attempts and waiting several minutes for a quota that resets later.
    if isinstance(exc, openai.RateLimitError):
        message = str(exc).lower()
        if "free-models-per-day" in message or "add 10 credits" in message or "daily limit" in message:
            return False
    return isinstance(exc, openai.APIStatusError) and exc.status_code in _RETRYABLE_STATUS


def chat_with_retry(client, *, attempts=6, base_delay=4.0, max_delay=60.0, log=print, **kwargs):
    """Call chat.completions.create, retrying transient failures with backoff.

    The client should be built with max_retries=0 so retries are counted here.
    Returns (response, attempts_used). Raises the last error if all attempts fail.
    """
    last = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            if not getattr(response, "choices", None):
                raise EmptyResponseError(f"Provider returned no choices: {getattr(response, 'error', None)}")
            return response, attempt
        except Exception as exc:  # noqa: BLE001 - classified below
            last = exc
            if attempt == attempts or not _is_retryable(exc):
                raise
            wait = _retry_after(exc)
            if wait is None:
                wait = min(max_delay, base_delay * 2 ** (attempt - 1))
                if isinstance(exc, openai.RateLimitError):
                    wait = max(wait, 15.0)  # per-minute limits need a real pause
            wait = min(max_delay, wait) + random.uniform(0, 1.5)
            log(f"    retry {attempt}/{attempts - 1} after {type(exc).__name__}; waiting {wait:.1f}s")
            time.sleep(wait)
    raise last  # pragma: no cover


def message_text(response):
    """Return (text, finish_reason). Falls back to reasoning text if content is empty."""
    choice = response.choices[0]
    message = choice.message
    text = (message.content or "").strip()
    if not text:
        text = (getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or "").strip()
    return text, getattr(choice, "finish_reason", None)


def extract_json_object(text, required_key="protocol"):
    """Pull the config object out of model text.

    Handles code fences, prose around the JSON, and reasoning text that
    contains several {...} fragments. Prefers the LAST object that has
    `required_key` (final answers come after the reasoning).
    """
    text = (text or "").strip()
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    found = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            found.append(value)
    for value in reversed(found):
        if required_key in value:
            return value
    if found:
        return found[-1]
    raise ValueError("Model response did not contain a JSON object")
