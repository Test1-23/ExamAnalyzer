import json
import re
import time
from typing import Optional

from openai import OpenAI

from .logger import get_logger

_log = get_logger()


from .constants import FLASH_MODEL, PRO_MODEL, DEFAULT_MAX_RETRIES


def create_client(api_url: str, api_key: str) -> OpenAI:
    """Create a shared OpenAI-compatible client."""
    return OpenAI(api_key=api_key, base_url=api_url)


def _create_debug_logger(debug_callback):
    """Return a logging function — uses debug_callback if provided, else print."""
    if debug_callback:
        return debug_callback
    return lambda msg: print(msg)


def _extract_json(text: str) -> dict:
    """Extract JSON from response text — handles both JSON mode and embedded JSON."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Regex fallback: balanced-brace matching for 1-level nested JSON
    for pattern in (r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    # Bracket-aware extraction: find the FIRST balanced { } or [ ] pair
    for open_c, close_c in [("{", "}"), ("[", "]")]:
        start = text.find(open_c)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_c:
                depth += 1
            elif text[i] == close_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


def _estimate_input_tokens(messages: list) -> int:
    """Rough token estimate: ~4 chars per token."""
    total = 0
    for m in messages:
        total += len(str(m.get("content", "")))
    return max(1, total // 4)


def _attempt_call(
    client: OpenAI,
    model: str,
    messages: list,
    response_format: Optional[dict] = None,
) -> str:
    """Single API call attempt. Returns raw response text on success, raises on failure."""
    t0 = time.time()
    input_tokens = _estimate_input_tokens(messages)
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        if model == PRO_MODEL:
            kwargs["reasoning_effort"] = "high"
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        if response_format:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from API")
        elapsed = int((time.time() - t0) * 1000)
        _log.debug(f"API call OK: model={model}, in={input_tokens}tok, out={len(content)//4}tok, {elapsed}ms")
        return content
    except Exception:
        elapsed = int((time.time() - t0) * 1000)
        _log.warning(f"API call FAIL: model={model}, in={input_tokens}tok, {elapsed}ms")
        raise


def _call_with_retry(
    client: OpenAI,
    model: str,
    messages: list,
    max_retries: int = DEFAULT_MAX_RETRIES,
    debug_callback=None,
    skip_json: bool = False,
) -> dict:
    """
    Universal retry wrapper.
    1st pass: try with JSON mode (skipped for Pro — thinking is incompatible).
    2nd pass: fall back to text mode if JSON mode failed.
    Returns parsed dict.
    """
    log = _create_debug_logger(debug_callback)
    fail_reason = ""
    total_attempts = 0

    json_modes = [False] if skip_json else [True, False]
    for use_json in json_modes:
        fmt = {"type": "json_object"} if use_json else None
        for attempt in range(max_retries):
            total_attempts += 1
            try:
                raw = _attempt_call(client, model, messages, response_format=fmt)
                result = _extract_json(raw)
                _log.info(f"API retry OK: model={model}, mode={'json' if use_json else 'text'}, attempts={total_attempts}")
                return result
            except Exception as e:
                fail_reason = str(e)
                wait = 2 ** attempt
                log(f"  {model} {'json' if use_json else 'text'} "
                    f"attempt {attempt + 1}/{max_retries} failed ({fail_reason}), "
                    f"retrying in {wait}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait)

    # Final fallback: simplify prompt — strip JSON requirement, shorten system prompt
    if fail_reason and ("Empty response" in fail_reason or "JSON" in fail_reason):
        log(f"  {model} attempting simplified prompt fallback...")
        try:
            simplified = [{"role": "system", "content": "Answer concisely."}]
            simplified.append({"role": "user", "content": messages[-1]["content"][:2000] if messages[-1]["role"] == "user" else str(messages[-1])})
            raw = _attempt_call(client, model, simplified, response_format=None)
            result = _extract_json(raw)
            _log.info(f"API retry OK: model={model}, mode=simplified_fallback, attempts={total_attempts + 1}")
            return result
        except Exception as e:
            _log.warning(f"  Simplified fallback also failed: {e}")

    _log.error(f"API retry EXHAUSTED: model={model}, attempts={total_attempts}, last_error={fail_reason}")
    raise RuntimeError(
        f"{model} failed after all retry strategies. Last error: {fail_reason}"
    )


# ---- Public API ----

def call_flash(
    client: OpenAI,
    messages: list,
    max_retries: int = DEFAULT_MAX_RETRIES,
    debug_callback=None,
) -> dict:
    """Call deepseek-v4-flash with retry + JSON fallback."""
    return _call_with_retry(client, FLASH_MODEL, messages, max_retries, debug_callback)
