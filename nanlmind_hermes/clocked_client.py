"""Scoped client for the NaNLMind_CLOCKED participant gateway (never the upstream SDK)."""
from __future__ import annotations

import json
import os
import re
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class ClockedGatewayError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def _endpoint(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ("https", "http") or not parsed.netloc:
        raise ValueError("CLOCKED_GATEWAY_URL must be an HTTP(S) URL")
    if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("CLOCKED gateway requires HTTPS except on localhost")
    return value.rstrip("/")


class ClockedGatewayClient:
    def __init__(self, base_url: str | None = None, token: str | None = None,
                 timeout: int = 90):
        self.base_url = _endpoint(base_url or os.environ.get("CLOCKED_GATEWAY_URL", ""))
        self.token = token if token is not None else os.environ.get("CLOCKED_GATEWAY_TOKEN", "")
        if not self.token or len(self.token) > 512:
            raise ValueError("CLOCKED_GATEWAY_TOKEN is missing or invalid")
        if not 1 <= timeout <= 300:
            raise ValueError("timeout must be between 1 and 300 seconds")
        self.timeout = timeout

    def health(self) -> dict:
        return self._request("GET", "/healthz")

    def execute(self, plan: dict, idempotency_key: str | None = None) -> dict:
        key = idempotency_key or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", key):
            raise ValueError("invalid idempotency key")
        body = {"protocol": "nanlm.clocked-task-plan", "version": 1, "plan": plan}
        return self._request("POST", "/v1/tasks", body, key)

    def _request(self, method: str, path: str, body: dict | None = None,
                 idempotency_key: str | None = None) -> dict:
        data = None if body is None else json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        headers = {"Authorization": "Bearer " + self.token, "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        req = Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read(1_048_577).decode("utf-8"))
        except HTTPError as exc:
            retryable = exc.code in (409, 429) or exc.code >= 500
            raise ClockedGatewayError("CLOCKED gateway returned HTTP %d" % exc.code,
                                      exc.code, retryable) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ClockedGatewayError("CLOCKED gateway transport failure", None, True) from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ClockedGatewayError("CLOCKED gateway returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise ClockedGatewayError("CLOCKED gateway response must be an object")
        return result
