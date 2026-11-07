#!/usr/bin/env python3
"""Opt-in live smoke test for a deployed NANNET authority and CLOCKED gateway.

This script contacts the endpoints you pass. It never prints tokens or pairing
codes. Create a one-time pairing code with the authority operator first, then:

    python scripts/live_smoke.py \
      --nannet-url https://<authority> \
      --pairing-code nnh_... \
      --clocked-url https://<gateway> \
      --clocked-token-file ~/.config/nanlmind/clocked-token \
      --output docs/LIVE_SMOKE.md

The script writes a JSON report next to the Markdown report. Live services can
change; record the exact output with the release commit.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import sys
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanlmind_hermes import auth_tools, hermes_secrets  # noqa: E402
from nanlmind_hermes.clocked_client import ClockedGatewayClient  # noqa: E402
from nanlmind_hermes.clocked_mode import execute_multi_source  # noqa: E402
from nanlmind_hermes.scratch import Scratch  # noqa: E402


class Context:
    def __init__(self, state_dir, nannet_url, issuer_key, clocked_url):
        self.config = {
            "state_dir": str(state_dir),
            "nannet_auth_url": nannet_url,
            "nannet_issuer_public_key": issuer_key,
            "clocked_gateway_url": clocked_url,
        }

    def get_config(self, name, default=None):
        return self.config.get(name, default)


def result(name, ok, detail):
    return {"check": name, "status": "pass" if ok else "fail", "detail": detail}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nannet-url", required=True)
    parser.add_argument("--pairing-code", required=True)
    parser.add_argument("--issuer-key", default="")
    parser.add_argument("--clocked-url", required=True)
    parser.add_argument("--clocked-token-file", required=True)
    parser.add_argument("--output", default=str(ROOT / "docs/LIVE_SMOKE.md"))
    args = parser.parse_args(argv)

    clocked_token = Path(args.clocked_token_file).read_text(encoding="utf-8").strip()
    if not clocked_token:
        raise SystemExit("clocked token file is empty")
    master = secrets.token_bytes(32)
    master_b64 = base64.urlsafe_b64encode(master).rstrip(b"=").decode()
    root = Path(tempfile.mkdtemp(prefix="nanlmind-live-smoke-"))
    issuer_key = args.issuer_key
    if not issuer_key:
        import urllib.request
        with urllib.request.urlopen(args.nannet_url.rstrip("/") + "/v1/issuer", timeout=15) as response:
            issuer_key = json.loads(response.read(32768))["issuer_public_key"]
    ctx = Context(root, args.nannet_url.rstrip("/"), issuer_key, args.clocked_url.rstrip("/"))
    secrets_map = {
        "NANNET_MASTER_KEY": master_b64,
        "CLOCKED_GATEWAY_TOKEN": clocked_token,
        "NANLM_DEVICE_TOKEN": "live-smoke-head-token",
    }
    checks = []
    with patch.object(hermes_secrets, "get_secret", side_effect=lambda n: secrets_map.get(n, "")), \
         patch.object(auth_tools, "get_secret", side_effect=lambda n: secrets_map.get(n, "")):
        init = auth_tools.auth_init(ctx)
        checks.append(result("nanlm_auth_init", True, {"installation_id": init["installation_id"]}))
        paired = auth_tools.auth_pair(ctx, args.pairing_code)
        checks.append(result("nanlm_auth_pair", paired["paired"],
                             {"scopes": paired["scopes"], "issuer_key_pinned": paired["issuer_key_pinned"]}))
        status = auth_tools.auth_status(ctx)
        checks.append(result("nanlm_auth_status", bool(status.get("token_valid")),
                             {"scopes": status.get("token_scopes")}))
        checks.append(result("nanlm_require_scope", auth_tools.require_scope(ctx, "training:execute"),
                             {"scope": "training:execute"}))

        client = ClockedGatewayClient(ctx.config["clocked_gateway_url"], clocked_token, timeout=120)
        plan = {
            "schema": "nanlm.clocked-task-plan/1", "version": 1,
            "plan_id": uuid.uuid4().hex,
            "tasks": [{
                "id": "live-pii", "source_id": "live-source", "endpoint": "pii",
                "text": "Contact live-smoke@example.com for the integration check.",
                "options": {}, "depends_on": [],
            }],
        }
        with tempfile.TemporaryDirectory(prefix="nanlmind-live-clocked-") as tmp:
            scratch = Scratch(tmp, "live")
            executed = execute_multi_source(client, scratch, "Live PII check",
                                            [{"id": "live-source", "text": "live"}], plan=plan)
            checks.append(result("nanlm_clocked_execute", executed["result_count"] == 1,
                                 {"result_count": executed["result_count"],
                                  "scratch_entries": executed["scratch"]["entry_count"]}))
            lanes = sorted({row["lane"] for row in executed["scratch"]["entries"]})
            checks.append(result("nanlm_clocked_scratch", "clocked_result" in lanes, {"lanes": lanes}))

        revoked = auth_tools.auth_revoke(ctx)
        checks.append(result("nanlm_auth_revoke", revoked["revoked"] and revoked["local_token_erased"],
                             {"revoked": revoked["revoked"],
                              "local_token_erased": revoked["local_token_erased"]}))

    passed = sum(1 for row in checks if row["status"] == "pass")
    report = {
        "schema": "nanlmind-hermes-live-smoke/1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nannet_url": args.nannet_url,
        "clocked_url": args.clocked_url,
        "checks": checks,
        "passed": passed,
        "failed": len(checks) - passed,
    }
    json_path = Path(args.output).with_name("live-smoke.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Live smoke results",
        "",
        "Generated by `scripts/live_smoke.py` against deployed services. Tokens and",
        "pairing codes are never written to this report.",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for row in checks:
        detail = json.dumps(row["detail"], sort_keys=True).replace("|", "\\|")
        lines.append("| `%s` | %s | %s |" % (row["check"], "PASS" if row["status"] == "pass" else "FAIL", detail))
    lines += ["", "**Summary:** %d/%d checks passed." % (passed, len(checks)), ""]
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": passed, "failed": len(checks) - passed, "output": args.output}, indent=2))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
