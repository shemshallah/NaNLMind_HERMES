"""HERMES client for the NaNLMind head-cortex receiving contract."""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class HeadCortexError(RuntimeError):
    pass


class HeadCortexClient:
    """Calls a NaNLMind receiving endpoint; never owns or mutates head state."""

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

    def process(self, package: dict) -> dict:
        if package.get("schema") != "nanlm.hermes-evidence-package":
            raise ValueError("unsupported evidence package")
        body = json.dumps(package, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
        request = Request(
            self.endpoint + "/v1/head-cortex/process", data=body,
            headers={"Authorization": "Bearer " + self.token,
                     "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read(4_194_305).decode("utf-8"))
        except HTTPError as exc:
            raise HeadCortexError("NaNLMind head cortex returned HTTP %d" % exc.code) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise HeadCortexError("NaNLMind head cortex is unavailable") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise HeadCortexError("NaNLMind head cortex returned invalid JSON") from exc
        if not isinstance(result, dict) or result.get("schema") != "nanlm.head-cortex-response":
            raise HeadCortexError("unsupported NaNLMind head-cortex response")
        return result
