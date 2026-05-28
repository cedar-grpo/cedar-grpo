"""
llm_client.py
-------------
LLM API wrapper with:
  - Per-request structured logging to JSONL files
  - In-memory caching to avoid redundant API calls across restarts
  - Clean JSON-mode responses (no regex parsing)
  - Support for OpenAI-compatible and Gemini structured output paths
"""

from __future__ import annotations

import datetime
import json
import os
import re
import threading
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

import config

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_client: OpenAI | None = None
_gemini_client: Any | None = None
_lock = threading.Lock()

# Cache: request_key -> parsed dict (populated from disk + live calls)
_response_cache: dict[tuple, dict] = {}
# Set of log-file paths already scanned into the cache
_loaded_logs: set[str] = set()
# Set of log-file paths already cleared this run
_cleared_logs: set[str] = set()


# ---------------------------------------------------------------------------
# Model-capability detection
# ---------------------------------------------------------------------------

def _is_modern_model(model: str) -> bool:
    """
    Return True for GPT-5 and newer models that require updated API parameters:
      - ``max_completion_tokens`` instead of ``max_tokens``
      - ``reasoning_effort`` instead of ``temperature``
      - ``"developer"`` role instead of ``"system"``
    Detection is based on the numeric major version in the model name
    (e.g. ``gpt-5-nano`` → 5 ≥ 5 → True; ``gpt-4o`` → 4 < 5 → False).
    """
    m = re.match(r"gpt-(\d+)", model.lower())
    if m:
        return int(m.group(1)) >= 5
    return False


def _is_gemini_model(model: str) -> bool:
    """Return True when the configured model should use Gemini SDK routing."""
    return model.strip().lower().startswith("gemini")


def get_client() -> OpenAI:
    """Return (and lazily initialise) the shared OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=config.API_TIMEOUT,
            max_retries=0,  # We handle retries ourselves
        )
    return _client


def get_gemini_client() -> Any:
    """Return (and lazily initialise) the shared Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        kwargs: dict[str, Any] = {"api_key": config.OPENAI_API_KEY}
        if config.GEMINI_BASE_URL:
            kwargs["http_options"] = {"base_url": config.GEMINI_BASE_URL}
        _gemini_client = genai.Client(**kwargs)
    return _gemini_client


def test_connection() -> bool:
    """Ping the API with a minimal request; return True if reachable."""
    print("\n[Testing API connection …]")
    model = config.JUDGE_MODEL
    try:
        if _is_gemini_model(model):
            from google.genai import types

            get_gemini_client().models.generate_content(
                model=model,
                contents="ping",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=5,
                ),
            )
            print(f"[OK] Connected – model '{model}' is accessible.")
            return True

        modern = _is_modern_model(model)
        kwargs: dict = dict(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
        )
        if modern:
            kwargs["max_completion_tokens"] = 5
            kwargs["reasoning_effort"] = config.REASONING_EFFORT
        else:
            kwargs["max_tokens"] = 5
            kwargs["temperature"] = 0.0
        get_client().chat.completions.create(**kwargs)
        print(f"[OK] Connected – model '{model}' is accessible.")
        return True
    except Exception as exc:
        print(f"[ERROR] API test failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return "N/A"


def _split_oss_channels(text: str | None) -> tuple[str | None, str | None]:
    """
    Some OSS models embed thinking content and final content in a single string
    separated by a special token.  Split them if present.
    """
    separator = "<|channel|>final<|message|>"
    if isinstance(text, str) and separator in text:
        thinking, final = text.split(separator, 1)
        return thinking.strip(), final.strip()
    return None, text


def _cache_key(
    run_id: str,
    checkpoint: str,
    dataset: str,
    problem_id: str,
    metric: str,
    model: str,
) -> tuple:
    return (_safe(run_id), _safe(checkpoint), _safe(dataset), _safe(problem_id), _safe(metric), _safe(model))


def _log_path(dataset: str) -> str:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    return os.path.join(config.LOG_DIR, f"{dataset}_llm_responses.jsonl")


def _load_cache_from_disk(log_path: str) -> None:
    """Populate the in-memory cache from an existing JSONL log file (once per path)."""
    if log_path in _loaded_logs:
        return
    _loaded_logs.add(log_path)

    if not os.path.exists(log_path):
        return

    try:
        with open(log_path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if obj.get("parse_status", "").startswith("success") and isinstance(obj.get("parsed_data"), dict):
                    key = _cache_key(
                        obj.get("run_id", ""),
                        obj.get("checkpoint", ""),
                        obj.get("dataset", ""),
                        obj.get("problem_id", ""),
                        obj.get("metric_type", ""),
                        obj.get("model", ""),
                    )
                    _response_cache[key] = obj["parsed_data"]
    except Exception:
        pass  # Corrupt log – skip silently


def _append_log(log_path: str, entry: dict) -> None:
    with _lock:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _openai_structured_call(
    *,
    model: str,
    input_messages: list[dict[str, str]],
    response_schema: type[BaseModel],
    max_completion_tokens: int,
) -> tuple[dict, str | None, str | None, dict[str, int | None], str | None]:
    """Run one OpenAI structured-output call."""
    modern = _is_modern_model(model)
    call_kwargs: dict[str, Any] = dict(
        model=model,
        messages=input_messages,
        response_format=response_schema,
    )
    if modern:
        call_kwargs["max_completion_tokens"] = max_completion_tokens
        call_kwargs["reasoning_effort"] = config.REASONING_EFFORT
    else:
        call_kwargs["max_tokens"] = max_completion_tokens
        call_kwargs["temperature"] = 0.0

    response = get_client().chat.completions.parse(**call_kwargs)
    msg = response.choices[0].message
    raw_content: str | None = getattr(msg, "content", None)
    raw_reasoning: str | None = getattr(msg, "reasoning_content", None)
    oss_thinking, _ = _split_oss_channels(raw_content)
    thinking = raw_reasoning or oss_thinking

    if msg.refusal:
        return {}, raw_content, thinking, {}, msg.refusal
    if not msg.parsed:
        raise ValueError("Empty or unparsed response")

    validated = msg.parsed.model_dump()
    usage = response.usage
    _details = getattr(usage, "completion_tokens_details", None)
    _prompt_details = getattr(usage, "prompt_tokens_details", None)
    tokens: dict[str, int | None] = {
        "input": getattr(usage, "prompt_tokens", None),
        "output": getattr(usage, "completion_tokens", None),
        "reasoning": getattr(_details, "reasoning_tokens", None),
        "cached_input": getattr(_prompt_details, "cached_tokens", None),
    }
    tokens = {k: v for k, v in tokens.items() if v is not None}
    return validated, raw_content, thinking, tokens, None


def _gemini_structured_call(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: type[BaseModel],
    max_completion_tokens: int,
) -> tuple[dict, str | None, str | None, dict[str, int | None], str | None]:
    """Run one Gemini structured-output call."""
    from google.genai import types

    response = get_gemini_client().models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.0,
            max_output_tokens=max_completion_tokens,
        ),
    )
    raw_content: str | None = getattr(response, "text", None)
    thinking: str | None = None
    if not raw_content:
        raise ValueError("Empty or unparsed response")

    parsed_obj = json.loads(raw_content)
    validated = response_schema.model_validate(parsed_obj).model_dump()
    usage = getattr(response, "usage_metadata", None)
    tokens: dict[str, int | None] = {
        "input": getattr(usage, "prompt_token_count", None) if usage is not None else None,
        "output": getattr(usage, "candidates_token_count", None) if usage is not None else None,
    }
    total_tokens = getattr(usage, "total_token_count", None) if usage is not None else None
    if total_tokens is not None:
        tokens["total"] = total_tokens
    tokens = {k: v for k, v in tokens.items() if v is not None}
    return validated, raw_content, thinking, tokens, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    source_full_input: str | None = None,
    response_schema: type[BaseModel],
    dataset: str = "unknown",
    problem_id: str = "N/A",
    metric_type: str = "unknown",
    checkpoint: str = "N/A",
    run_id: str | None = None,
) -> dict:
    """
    Call the judge LLM and return a plain dict matching `response_schema`.

    The response is requested in JSON mode and validated against the Pydantic
    schema.  All calls are logged to JSONL and cached so that restarting the
    script does not re-bill completed items.

    Returns a dict on success.  Returns the schema's ``default()`` values on
    any irrecoverable failure so the pipeline can continue.
    """
    if run_id is None:
        run_id = config.RUN_ID

    model = config.JUDGE_MODEL
    log_path = _log_path(dataset)

    # Optional per-run log reset
    if config.CLEAR_PREVIOUS_OUTPUTS and log_path not in _cleared_logs:
        try:
            os.remove(log_path)
        except FileNotFoundError:
            pass
        _cleared_logs.add(log_path)

    _load_cache_from_disk(log_path)

    key = _cache_key(run_id, checkpoint, dataset, problem_id, metric_type, model)
    if key in _response_cache:
        print(f"  [LLM] Cache hit – {metric_type}/{problem_id} (ckpt={checkpoint})")
        return _response_cache[key]

    # All models honour API_MAX_RETRIES; transient errors are always worth retrying
    max_tries = max(1, config.API_MAX_RETRIES)

    effective_max_tokens = config.METRIC_MAX_COMPLETION_TOKENS.get(
        metric_type, config.MAX_COMPLETION_TOKENS
    )

    modern = _is_modern_model(model)
    is_gemini = _is_gemini_model(model)
    system_role = "developer" if modern else "system"
    input_messages = [
        {"role": system_role, "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: str = ""
    raw_question = source_full_input if source_full_input is not None else ""
    for attempt in range(max_tries):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        try:
            if is_gemini:
                validated, raw_content, thinking, tokens, refusal = _gemini_structured_call(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_schema=response_schema,
                    max_completion_tokens=effective_max_tokens,
                )
            else:
                validated, raw_content, thinking, tokens, refusal = _openai_structured_call(
                    model=model,
                    input_messages=input_messages,
                    response_schema=response_schema,
                    max_completion_tokens=effective_max_tokens,
                )

            if refusal:
                _append_log(log_path, {
                    "timestamp": ts, "run_id": run_id, "checkpoint": checkpoint,
                    "dataset": dataset, "problem_id": problem_id,
                    "metric_type": metric_type, "model": model,
                    "parse_status": "refusal", "error_message": refusal,
                    "raw_question": raw_question,
                    "raw_response": raw_content, "raw_reasoning": thinking,
                    "user_prompt": user_prompt,
                    "input_messages": input_messages,
                })
                return _default_payload(response_schema)

            _append_log(log_path, {
                "timestamp": ts, "run_id": run_id, "checkpoint": checkpoint,
                "dataset": dataset, "problem_id": problem_id,
                "metric_type": metric_type, "model": model,
                "parse_status": "success",
                "tokens": tokens,
                "raw_question": raw_question,
                "raw_response": raw_content,
                "raw_reasoning": thinking,
                "user_prompt": user_prompt,
                "input_messages": input_messages,
                "parsed_data": validated,
            })
            # Attach token info as a side-channel key so callers can read it
            # without it polluting the schema-validated fields.
            validated["__tokens__"] = tokens
            _response_cache[key] = validated
            return validated

        except (ValueError, KeyError) as parse_err:
            last_error = str(parse_err)
            if attempt < max_tries - 1:
                print(f"  [LLM] JSON parse error (attempt {attempt + 1}/{max_tries}): {last_error[:120]} – retrying …")
                continue

        except Exception as api_err:
            last_error = str(api_err)
            err_lower = last_error.lower()
            if attempt < max_tries - 1:
                print(f"  [LLM] API error (attempt {attempt + 1}/{max_tries}): {last_error[:120]} – retrying …")
                continue
            # Surface useful hints
            if "timeout" in err_lower:
                print("  [HINT] Increase API_TIMEOUT in config.py or check network.")
            elif "not found" in err_lower or "404" in err_lower:
                if is_gemini:
                    print(f"  [HINT] Model '{model}' not found – check JUDGE_MODEL and GEMINI_BASE_URL in config.py/.env.")
                else:
                    print(f"  [HINT] Model '{model}' not found – check JUDGE_MODEL in config.py.")

    _append_log(log_path, {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "run_id": run_id, "checkpoint": checkpoint,
        "dataset": dataset, "problem_id": problem_id,
        "metric_type": metric_type, "model": model,
        "parse_status": "failed", "error_message": last_error,
        "raw_question": raw_question,
        "raw_response": None, "raw_reasoning": None,
        "user_prompt": user_prompt,
        "input_messages": input_messages,
    })
    print(f"  [LLM] FAILED {metric_type}/{problem_id} ckpt={checkpoint} after {max_tries} attempt(s): {last_error[:200]}")
    return _default_payload(response_schema)



def _default_payload(schema: type[BaseModel]) -> dict:
    """Return a zeroed-out dict for the given Pydantic schema."""
    defaults: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        # Simple type checking
        if str(field.annotation) == "bool":
            defaults[name] = False
        elif str(field.annotation) == "str":
            defaults[name] = ""
        elif "list" in str(field.annotation).lower():
            defaults[name] = []
        else:
            defaults[name] = None
    try:
        return schema(**defaults).model_dump()
    except Exception:
        return defaults
