"""HERMES client for the NaNLMind head-cortex receiving contract."""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class HeadCortexError(RuntimeError):
    """A transport or contract failure from the NaNLMind head-cortex service."""


class HeadCortexClient:
    """Calls NaNLMind receiving endpoints; never owns or mutates canonical head state."""

    def __init__(self, endpoint: str | None = None, token: str | None = None,
                 timeout: int = 90):
        endpoint = endpoint or os.environ.get("NANLM_HEAD_CORTEX_URL", "")
        parsed = urlparse(endpoint)
        if parsed.scheme not in ("https", "http") or not parsed.netloc:
            raise ValueError("NANLM_HEAD_CORTEX_URL must be an HTTP(S) URL")
        if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("head cortex requires HTTPS except on localhost")
        self.endpoint = endpoint.rstrip("/")
        self.token = token if token is not None else os.environ.get("NANLM_DEVICE_TOKEN", "")
        if not self.token or len(self.token) > 512:
            raise ValueError("NANLM_DEVICE_TOKEN is missing or invalid")
        if not 1 <= timeout <= 300:
            raise ValueError("timeout must be between 1 and 300 seconds")
        self.timeout = timeout

    def _post(self, path: str, payload: dict, request_schema: str,
              response_schema: str, max_bytes: int,
              require_non_promotion: bool = False) -> dict:
        if not isinstance(payload, dict) or payload.get("schema") != request_schema:
            raise ValueError("unsupported request schema: " + request_schema)
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
        if len(body) > max_bytes:
            raise ValueError("head-cortex request exceeds configured size limit")
        call = Request(
            self.endpoint + path, data=body,
            headers={"Authorization": "Bearer " + self.token,
                     "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(call, timeout=self.timeout) as response:
                raw = response.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    raise HeadCortexError("head-cortex response exceeds configured size limit")
                result = json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            raise HeadCortexError("NaNLMind head cortex returned HTTP %d" % exc.code) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise HeadCortexError("NaNLMind head cortex is unavailable") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise HeadCortexError("NaNLMind head cortex returned invalid JSON") from exc
        if not isinstance(result, dict) or result.get("schema") != response_schema:
            raise HeadCortexError("unsupported NaNLMind response schema: " + response_schema)
        if require_non_promotion and (
            result.get("promotion_allowed") is not False
            or result.get("canonical_state_modified") is not False
        ):
            raise HeadCortexError("candidate evaluation violated non-promotion contract")
        return result

    def process(self, package: dict) -> dict:
        return self._post(
            "/v1/head-cortex/process", package,
            "nanlm.hermes-evidence-package", "nanlm.head-cortex-response",
            4_194_304,
        )

    def create_candidate(self, request: dict) -> dict:
        return self._post(
            "/v1/head-cortex/candidate", request,
            "nanlm.head-candidate-request/1", "nanlm.head-candidate-response/1",
            65_536,
        )

    def evaluate_candidate(self, request: dict) -> dict:
        return self._post(
            "/v1/head-cortex/candidate/evaluate", request,
            "nanlm.head-candidate-evaluation-request/1",
            "nanlm.head-candidate-evaluation-response/1", 1_048_576,
            require_non_promotion=True,
        )
