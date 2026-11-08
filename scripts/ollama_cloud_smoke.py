#!/usr/bin/env python3
"""One bounded, authenticated Ollama Gemma 4 cloud request (explicit opt-in).

Never pass an Ollama key as an argument or in a source file. This is host-side
HTTP transport only; it does not perform NaN16DE routing/model computation.
Ollama cloud is metered by tokens. A successful request does NOT prove that
usage was included in a free account's credits; check the billing/usage UI.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

API_ENDPOINT = "https://ollama.com/v1/chat/completions"
MODEL = "gemma4:31b"
MAX_OUTPUT_TOKENS = 24
PROMPT = "Return exactly one word: HELLO"


def smoke(api_key: str, open_url=None) -> dict:
    if not api_key or len(api_key) > 1024 or any(c in api_key for c in "\r\n\x00"):
        raise ValueError("valid OLLAMA_API_KEY environment secret required")
    body = json.dumps({
        "model": MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
    }, separators=(",", ":")).encode()
    request = urllib.request.Request(
        API_ENDPOINT, data=body,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    caller = open_url or urllib.request.urlopen
    try:
        with caller(request, timeout=60) as response:
            raw = response.read(32769)
            if len(raw) > 32768:
                raise RuntimeError("Ollama response exceeds bounded size")
    except urllib.error.HTTPError as exc:
        # Never reflect untrusted bodies or headers containing user secrets.
        if exc.code in (401, 403):
            raise RuntimeError("Ollama rejected API key or denied model access (HTTP %d)" % exc.code) from None
        if exc.code in (402, 429):
            raise RuntimeError("Ollama refused metered credit/quota request (HTTP %d)" % exc.code) from None
        raise RuntimeError("Ollama cloud request failed (HTTP %d)" % exc.code) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError("Ollama cloud API is unreachable from this host") from None
    try:
        value = json.loads(raw)
        answer = value["choices"][0]["message"].get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("empty response")
        usage = value.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        if (not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool)
                or not isinstance(output_tokens, int) or isinstance(output_tokens, bool)
                or prompt_tokens < 0 or output_tokens < 0):
            raise ValueError("invalid usage counters")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("Ollama response lacks a valid completion") from None

    # Official list rates: 0.14 USD / 1M input, 0.40 USD / 1M output.
    # Estimate is informational; actual bills depend on cached rates/credits.
    return {
        "pass": True,
        "model_requested": MODEL,
        "response_nonempty": True,
        "response_characters": len(answer),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "estimated_usd_at_list_rate": round(
            (prompt_tokens * 0.14 + output_tokens * 0.40) / 1_000_000, 8
        ),
        "included_in_free_credits": "not_determinable_from_inference_response",
        "uses_real_cloud_inference": True,
    }


if __name__ == "__main__":
    if os.environ.get("OLLAMA_CLOUD_SMOKE_OPT_IN") != "yes":
        print("Set OLLAMA_CLOUD_SMOKE_OPT_IN=yes to authorize one metered request.",
              file=sys.stderr)
        raise SystemExit(2)
    try:
        report = smoke(os.environ.get("OLLAMA_API_KEY", ""))
        print(json.dumps(report, sort_keys=True, indent=2))
    except (RuntimeError, ValueError) as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
