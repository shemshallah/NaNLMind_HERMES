"""HERMES client for the NaNLMind head-cortex receiving contract."""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class HeadCortexError(RuntimeError):
    pass


class HeadCortexClient:
    """Calls a NaNLMind receiving endpoint; never owns or mutates head state."""

    def __init__(self, endpoint: str | None = None, token: str | None = None,
                 timeout: int = 90):
        endpoint = endpoint or ""
        parsed = urlparse(endpoint)
        if parsed.scheme not in ("https", "http") or not parsed.netloc:
            raise ValueError("NANLM_HEAD_CORTEX_URL must be an HTTP(S) URL")
        if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("head cortex requires HTTPS except on localhost")
        self.endpoint = endpoint.rstrip("/")
        self.token = token or ""
        if not self.token or len(self.token) > 4096:
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

    def create_candidate(self, request: dict) -> dict:
        if not isinstance(request, dict) or request.get("schema") != "nanlm.head-candidate-request/1":
            raise ValueError("unsupported head-candidate request")
        body = json.dumps(request, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
        call = Request(
            self.endpoint + "/v1/head-cortex/candidate", data=body,
            headers={"Authorization": "Bearer " + self.token,
                     "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(call, timeout=self.timeout) as response:
                result = json.loads(response.read(65537).decode("utf-8"))
        except HTTPError as exc:
            raise HeadCortexError("NaNLMind candidate endpoint returned HTTP %d" % exc.code) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise HeadCortexError("NaNLMind candidate endpoint is unavailable") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise HeadCortexError("NaNLMind candidate endpoint returned invalid JSON") from exc
        if not isinstance(result, dict) or result.get("schema") != "nanlm.head-candidate-response/1":
            raise HeadCortexError("unsupported NaNLMind candidate response")
        return result

    def evaluate_candidate(self, request: dict) -> dict:
        """Evaluate a candidate on held-out rows without permitting promotion or mutation."""
        if not isinstance(request, dict) or request.get("schema") != "nanlm.head-candidate-evaluation-request/1":
            raise ValueError("unsupported head-candidate evaluation request")
        body = json.dumps(request, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
        if len(body) > 1_048_576:
            raise ValueError("candidate evaluation request exceeds 1 MiB")
        call = Request(
            self.endpoint + "/v1/head-cortex/candidate/evaluate", data=body,
            headers={"Authorization": "Bearer " + self.token,
                     "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(call, timeout=self.timeout) as response:
                raw = response.read(1_048_577)
                if len(raw) > 1_048_576:
                    raise HeadCortexError("candidate evaluation response exceeds 1 MiB")
                result = json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            raise HeadCortexError("NaNLMind candidate evaluation returned HTTP %d" % exc.code) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise HeadCortexError("NaNLMind candidate evaluation is unavailable") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise HeadCortexError("NaNLMind candidate evaluation returned invalid JSON") from exc
        if not isinstance(result, dict) or result.get("schema") != "nanlm.head-candidate-evaluation-response/1":
            raise HeadCortexError("unsupported NaNLMind candidate evaluation response")
        if result.get("promotion_allowed") is not False or result.get("canonical_state_modified") is not False:
            raise HeadCortexError("candidate evaluation violated non-promotion contract")
        return result

    def create_learning_candidate(self, request: dict) -> dict:
        if not isinstance(request, dict) or request.get("schema") != "nanlm.head-learning-candidate-request/1":
            raise ValueError("unsupported head-learning candidate request")
        if request.get("teacher_outputs_are_ground_truth") is not False or request.get("split") != "train":
            raise ValueError("learning request must mark CLOCKED as weak supervision on the train split")
        digest = request.get("base_topology_digest")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest.lower())):
            raise ValueError("learning request requires a hexadecimal base topology digest")
        body = json.dumps(request, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
        if len(body) > 1_048_576:
            raise ValueError("learning candidate request exceeds 1 MiB")
        call = Request(
            self.endpoint + "/v1/head-cortex/candidate/learn", data=body,
            headers={"Authorization": "Bearer " + self.token,
                     "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(call, timeout=self.timeout) as response:
                raw = response.read(1_048_577)
                if len(raw) > 1_048_576:
                    raise HeadCortexError("learning candidate response exceeds 1 MiB")
                result = json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            raise HeadCortexError("NaNLMind learning candidate returned HTTP %d" % exc.code) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise HeadCortexError("NaNLMind learning candidate endpoint is unavailable") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise HeadCortexError("NaNLMind learning candidate returned invalid JSON") from exc
        if not isinstance(result, dict) or result.get("schema") != "nanlm.head-learning-candidate-response/1":
            raise HeadCortexError("unsupported NaNLMind learning candidate response")
        if (result.get("teacher_outputs_are_ground_truth") is not False
                or result.get("promotion_allowed") is not False
                or result.get("canonical_state_modified") is not False):
            raise HeadCortexError("learning candidate violated weak-supervision isolation policy")
        return result
