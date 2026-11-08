#!/usr/bin/env python3
"""Run the 1.3.2 demo against every registered tool and write a results report.

The demo is local and deterministic. It starts a real NANNET pairing authority
over loopback HTTP, local CLOCKED and head-cortex stubs, and uses the real
``nanlm-universal`` binary for native training. No production credentials or
network services are contacted.

Usage:
    python scripts/demo_run.py --native-binary /path/to/nanlm-universal
    python scripts/demo_run.py --native-binary ... --output docs/DEMO_RESULTS.md

The JSON report is written next to the Markdown report as ``demo-results.json``.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import sys
import tempfile
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanlmind_hermes import register  # noqa: E402
from nanlmind_hermes.auth import generate_master_key  # noqa: E402
from nanlmind_hermes.http_api import make_server as make_authority  # noqa: E402
from nanlmind_hermes.pairing import PairingAuthority  # noqa: E402
from nanlmind_hermes import hermes_secrets  # noqa: E402
from nanlmind_hermes import auth_tools, clocked_tools, gemma_teacher  # noqa: E402

ZERO = "0" * 64


class DemoLLM:
    """Deterministic provider used only by the demo."""

    def complete(self, messages, **kwargs):
        purpose = kwargs.get("purpose", "")
        system = messages[0].get("content", "") if messages else ""
        user = messages[1].get("content", "") if len(messages) > 1 else ""
        if purpose == "nanlmind-hermes-multiresolution-reflection":
            text = json.dumps({"discoveries": []})
        elif purpose == "nanlmind-hermes-chat":
            text = "NaNLMind demo reply. This response is local, deterministic, and not a model benchmark."
        elif purpose.startswith("nanlmind-relation-teacher-"):
            text = json.dumps({"records": [
                {"subject": f"demo:{purpose[-3:]}", "predicate": "supports",
                 "object": "The river flows.", "polarity": "supports",
                 "confidence": 0.9, "evidence": "demo fixture"},
                {"subject": f"demo:{purpose[-3:]}", "predicate": "refutes",
                 "object": "The river is dry.", "polarity": "refutes",
                 "confidence": 0.8, "evidence": "demo fixture"},
                {"subject": f"demo:{purpose[-3:]}", "predicate": "supports",
                 "object": "Birds fly above the water.", "polarity": "supports",
                 "confidence": 0.85, "evidence": "demo fixture"},
                {"subject": f"demo:{purpose[-3:]}", "predicate": "refutes",
                 "object": "Boats move upstream.", "polarity": "refutes",
                 "confidence": 0.75, "evidence": "demo fixture"},
                {"subject": f"demo:{purpose[-3:]}", "predicate": "supports",
                 "object": "The sun sets behind the hills.", "polarity": "supports",
                 "confidence": 0.8, "evidence": "demo fixture"},
                {"subject": f"demo:{purpose[-3:]}", "predicate": "refutes",
                 "object": "The valley is empty.", "polarity": "refutes",
                 "confidence": 0.7, "evidence": "demo fixture"},
            ]})
        elif purpose == "nanlmind-corpus-generation":
            text = "\n".join(
                f"demo corpus record {i}: the river flows through the valley and birds fly above the water"
                for i in range(1, 65))
        elif purpose.startswith("nanlmind-curriculum-") or "Generate training examples" in system:
            text = "\n".join(f"demo curriculum record {i}: the river flows through the valley"
                             for i in range(1, 65))
        elif purpose == "nanlm-hermes-clocked-mode-selection":
            if "PII" in user or "privacy" in user.lower():
                text = json.dumps({"mode": "CLOCKED_PRIVACY_AUDIT", "reason": "demo privacy audit"})
            elif "compare" in user.lower():
                text = json.dumps({"mode": "CLOCKED_COMPARE", "reason": "demo comparison"})
            else:
                text = json.dumps({"mode": "CLOCKED_EXTRACT", "reason": "demo extraction"})
        elif purpose == "nanlm-hermes-clocked-synthesis":
            text = "Demo synthesis: the sources support the river-flow claim; unresolved conflicts are marked."
        elif purpose == "nanlm-hermes-final-synthesis":
            text = "Demo final answer with evidence attribution and explicit uncertainty."
        else:
            text = "Demo fixture response."
        return SimpleNamespace(text=text, provider="demo-provider", model="demo-model")


class DemoContext:
    def __init__(self, config):
        self.config = config
        self.llm = DemoLLM()
        self.tools = {}
        self.command = None

    def get_config(self, name, default=None):
        return self.config.get(name, default)

    def register_tool(self, name, toolset=None, handler=None, schema=None):
        self.tools[name] = handler

    def register_command(self, name, handler, description=None):
        self.command = handler


class JsonHandler(BaseHTTPRequestHandler):
    server_version = "NaNLMindDemo/1"

    def log_message(self, fmt, *args):
        return

    def reply(self, status, payload):
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        size = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(size)) if size else {}


class ClockedStub(JsonHandler):
    def do_POST(self):
        if self.path != "/v1/tasks":
            return self.reply(404, {"error": "not_found"})
        payload = self.body()
        plan = payload.get("plan", {})
        results = []
        for task in plan.get("tasks", []):
            endpoint = task.get("endpoint")
            fields = {}
            if endpoint == "pii":
                fields = {"detected": False, "values": []}
            elif endpoint == "extract":
                fields = {"claims": ["The river flows.", "Birds fly above the water."]}
            elif endpoint == "grounding":
                fields = {"verdict": "supported"}
            elif endpoint == "compare":
                fields = {"agreements": ["river-flow"], "conflicts": []}
            elif endpoint == "decide":
                fields = {"decision": "supported", "reason": "demo evidence"}
            results.append({
                "task_id": task.get("id"), "source_id": task.get("source_id"),
                "endpoint": endpoint, "result_type": endpoint, "status": "ok",
                "provenance": {"input_sha256": hashlib.sha256(
                    str(task.get("text", "")).encode()).hexdigest()},
                "result": {"fields": fields},
            })
        self.reply(200, {"protocol": "nanlm.clocked-task-results", "version": 1,
                         "plan_id": plan.get("plan_id"), "results": results})


class HeadStub(JsonHandler):
    def do_POST(self):
        if self.path == "/v1/head-cortex/process":
            package = self.body()
            return self.reply(200, {
                "schema": "nanlm.head-cortex-response",
                "request_sha256": package.get("package_sha256"),
                "generation": 1,
                "truth_state": "TRUE",
                "relations": [{"truth_state": "TRUE", "head_id": 0, "premises": []}],
                "abstentions": [],
                "provenance": {"read_only": True, "state_digest": "a" * 64,
                               "source_entry_ids": []},
            })
        if self.path == "/v1/head-cortex/candidate":
            return self.reply(200, {
                "schema": "nanlm.head-candidate-response/1",
                "status": "candidate_only", "candidate_digest": "b" * 64,
                "canonical_state_modified": False, "promotion_allowed": False,
            })
        if self.path == "/v1/head-cortex/candidate/evaluate":
            return self.reply(200, {
                "schema": "nanlm.head-candidate-evaluation-response/1",
                "status": "evaluated_candidate", "utility_delta": 0.1,
                "recommendation": "review_candidate",
                "baseline": {"accuracy": 0.5}, "candidate": {"accuracy": 0.6},
                "promotion_allowed": False, "canonical_state_modified": False,
            })
        if self.path == "/v1/head-cortex/candidate/learn":
            return self.reply(200, {
                "schema": "nanlm.head-learning-candidate-response/1",
                "status": "candidate_only", "candidate_digest": "c" * 64,
                "teacher_outputs_are_ground_truth": False,
                "promotion_allowed": False, "canonical_state_modified": False,
            })
        return self.reply(404, {"error": "not_found"})


def start_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, "http://127.0.0.1:%d" % server.server_address[1]


def summarize(value, limit=220):
    text = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "..."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-binary", default=str(Path.home() / "NaNLMind/build/nanlm-universal"))
    parser.add_argument("--output", default=str(ROOT / "docs/DEMO_RESULTS.md"))
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    native = Path(args.native_binary).expanduser()
    if not native.is_file():
        raise SystemExit("native binary not found: %s" % native)

    root = Path(tempfile.mkdtemp(prefix="nanlmind-demo-"))
    master = secrets.token_bytes(32)
    master_b64 = base64.urlsafe_b64encode(master).rstrip(b"=").decode()
    authority = PairingAuthority(root / "authority.sqlite", master)

    # Use the plugin's real stdlib authority for pairing/authorize/revoke.
    authority_http = make_authority(authority, "127.0.0.1", 0)
    authority_thread = threading.Thread(target=authority_http.serve_forever, daemon=True)
    authority_thread.start()
    authority_url = "http://127.0.0.1:%d" % authority_http.server_address[1]
    clocked_server, clocked_thread, clocked_url = start_server(ClockedStub)
    head_server, head_thread, head_url = start_server(HeadStub)

    secrets_map = {
        "NANNET_MASTER_KEY": master_b64,
        "CLOCKED_GATEWAY_TOKEN": "clu_demo_participant_token",
        "NANLM_DEVICE_TOKEN": "demo_device_token",
    }
    config = {
        "state_dir": str(root / "state"),
        "runner_command": [str(native)],
        "nannet_auth_url": authority_url,
        "nannet_issuer_public_key": authority.issuer_public_key(),
        "clocked_gateway_url": clocked_url,
        "nanlm_head_cortex_url": head_url,
        "chat_learning": True,
    }
    code = authority.create_pairing_code(
        ["universe:read", "training:execute", "corpus:write"], ttl_seconds=600)["pairing_code"]
    wordnet_path = root / "demo-wordnet.zip"
    with zipfile.ZipFile(wordnet_path, "w") as archive:
        archive.writestr(
            "wn3.1.dict/dict/data.noun",
            "\n".join([
                "00001740 03 n 01 entity 0 0 0 0 0 0 | something that exists",
                "00001741 03 n 01 physical_entity 0 0 1 @ 00001740 n 0000 | an entity with physical form",
                "00001742 03 n 02 copper 0 metal 0 2 @ 00001741 n 0000 #m 00001741 n 0000 | a reddish metal that conducts electricity",
            ]) + "\n",
        )
        for name in ("data.verb", "data.adj", "data.adv"):
            archive.writestr("wn3.1.dict/dict/" + name, "")
    config["wordnet_archive"] = str(wordnet_path)
    config["learning_min_support"] = 1
    config["learning_require_approval"] = False
    ctx = DemoContext(config)
    register(ctx)

    results = []

    def sanitize(text):
        return (text.replace(str(root), "<demo-state>")
                    .replace(str(native), "nanlm-universal"))

    def call(name, params=None):
        started = time.monotonic()
        try:
            raw = ctx.tools[name](params or {})
            value = json.loads(raw) if isinstance(raw, str) else raw
            item = {"tool": name, "status": "pass", "ms": round((time.monotonic() - started) * 1000, 1),
                    "summary": sanitize(summarize(value))}
        except Exception as exc:
            item = {"tool": name, "status": "fail", "ms": round((time.monotonic() - started) * 1000, 1),
                    "summary": sanitize("%s: %s" % (type(exc).__name__, str(exc)[:300]))}
        results.append(item)
        print(("PASS" if item["status"] == "pass" else "FAIL"), name, item["summary"][:160])
        return item, (value if item["status"] == "pass" else None)

    with patch.object(hermes_secrets, "get_secret", side_effect=lambda n: secrets_map.get(n, "")), \
         patch.object(auth_tools, "get_secret", side_effect=lambda n: secrets_map.get(n, "")), \
         patch.object(clocked_tools, "get_secret", side_effect=lambda n: secrets_map.get(n, "")):
        # Identity and NANNET.
        call("nanlm_auth_init")
        call("nanlm_auth_status")
        call("nanlm_auth_pair", {"pairing_code": code})
        call("nanlm_auth_status")

        # Android-compatible NANNET HW hardware-node lifecycle is a separate
        # encrypted login, not a HERMES native-universe authorization. Use
        # a deterministic loopback stub, never production endpoint tokens.
        from nanlmind_hermes import nannet_device_client as _hw
        def fake_hw_http(context, endpoint, body=None, token=""):
            identity = {"address": "nan://GM0:SC0:MC0:BS0:HW5",
                        "node_point": 6, "lease_id": "demo-hw-lease",
                        "lease_expires_at": 2000000000}
            if endpoint == "/v1/devices/register":
                assert body["consent"] is True
                return {**identity, "device_id": "demo-hw-device",
                        "token": "nnt_demo_hw"}
            if endpoint == "/v1/devices/rotate":
                assert token.startswith("nnt_")
                return {**identity, "token": "nnt_rotated_hw"}
            if endpoint == "/v1/devices/heartbeat":
                assert token == "nnt_demo_hw"
                return identity
            if endpoint == "/v1/devices/report":
                return {"accepted": True, "report_id": "demo-report"}
            if endpoint == "/v1/devices/release":
                return {"released": True, "address": identity["address"]}
            raise AssertionError(endpoint)
        with patch.object(_hw, "_http_json", side_effect=fake_hw_http):
            call("nanlm_hw_register", {"consent": True})
            call("nanlm_hw_status")
            call("nanlm_hw_heartbeat")
            call("nanlm_hw_report", {"report_type": "network_error",
                                     "summary": "Offline deterministic demo"})
            call("nanlm_hw_release")

        # Deterministic independent-universe *router receipt* fixture.
        # These calls exercise the real Hermes tool handlers and local
        # encryption, not a live router, guest execution, or canonical merge.
        from nanlmind_hermes import router_universe_client as _leased
        installation_id = auth_tools.auth_status(ctx)["installation_id"]
        fixture = {
            "schema": "nanrouter.hermes-lease.v1",
            "issuer": "nanrouter/BS0:0", "network_port": "BS0:0",
            "hardware_root": "HW0:0", "installation_id": installation_id,
            "slot": 0, "lease_id": 101, "universe_id": 501,
            "generation": 1, "revision": 1, "owner_tag": 97,
            "node_point": 3, "expires_at": int(time.time()) + 900,
            "heads": 255, "nannet_port_head": 255, "receipt_mac": "a" * 64,
        }
        def fake_router_lease(context, action, request):
            if action == "open":
                return installation_id, {"receipt": dict(fixture)}
            if action == "resolve":
                assert request["head_index"] == 254
                return installation_id, {"route": {
                    "admitted": True, "slot": fixture["slot"],
                    "lease_id": fixture["lease_id"],
                    "universe_id": fixture["universe_id"],
                    "generation": fixture["generation"],
                    "owner_tag": fixture["owner_tag"],
                    "node_point": fixture["node_point"],
                    "expires_at": fixture["expires_at"],
                    "network_port": "BS0:0", "head_index": 254,
                    "native_execution_committed": False,
                }}
            if action == "dispatch":
                frame = bytes.fromhex(request["frame_hex"])
                return installation_id, {
                    "dispatch_id": request["dispatch_id"], "head_index": request["head_index"],
                    "lease_id": fixture["lease_id"], "universe_id": fixture["universe_id"],
                    "native_guest_route_admitted": True, "n16t_native_guest_exchange": True,
                    "frame_sha256": hashlib.sha256(frame).hexdigest(),
                    "response_frame_hex": frame.hex(),
                    "response_sha256": hashlib.sha256(frame).hexdigest(),
                    "acknowledged": False, "replayed_from_journal": False,
                    "model_training_committed": False,
                }
            if action == "ack":
                return installation_id, {
                    "acknowledged": True, "dispatch_id": request["dispatch_id"],
                    "response_sha256": request["response_sha256"],
                    "model_training_committed": False,
                }
            if action == "inspect":
                response = b"N16T" + bytes(32)
                return installation_id, {
                    "dispatch_id": request["dispatch_id"],
                    "head_index": request["head_index"],
                    "lease_id": fixture["lease_id"],
                    "universe_id": fixture["universe_id"],
                    "state": "acknowledged",
                    "response_frame_hex": response.hex(),
                    "response_sha256": hashlib.sha256(response).hexdigest(),
                    "n16t_native_guest_exchange": True,
                    "model_training_committed": False,
                }
            if action == "math":
                assert request["kernel_id"] == "mth.u32.add_checked"
                assert request["a"] == 7 and request["b"] == 9
                return installation_id, {
                    "kernel_id": request["kernel_id"], "head_index": request["head_index"],
                    "value_u32": 16, "exact": True,
                    "native_guest_route_admitted": True,
                    "native_guest_universe_id": fixture["universe_id"],
                    "native_guest_node_point": fixture["node_point"],
                    "node_point": fixture["node_point"],
                    "universe_id": fixture["universe_id"],
                    "native_nan16_execution": True,
                    "model_training_committed": False,
                }
            if action == "renew":
                fixture["revision"] += 1
                fixture["expires_at"] += 900
                return installation_id, {"receipt": dict(fixture)}
            if action == "release":
                return installation_id, {
                    "released": True, "lease_id": fixture["lease_id"]}
            raise AssertionError(action)
        with patch.object(_leased, "_post", side_effect=fake_router_lease):
            call("nanlm_universe_status")
            call("nanlm_universe_open", {"alias": "demo"})
            call("nanlm_universe_resolve", {"alias": "demo", "head_index": 254})
            _, dispatched = call("nanlm_universe_dispatch", {
                "alias": "demo", "head_index": 254,
                "dispatch_id": "demo-n16t-dispatch-0001",
                "frame_hex": (b"N16T" + bytes(32)).hex()})
            call("nanlm_universe_ack", {
                "alias": "demo", "head_index": 254,
                "dispatch_id": "demo-n16t-dispatch-0001",
                "response_sha256": (dispatched or {}).get("response_sha256", "0" * 64)})
            call("nanlm_universe_inspect", {
                "alias": "demo", "head_index": 254,
                "dispatch_id": "demo-n16t-dispatch-0001"})
            call("nanlm_native_mth_u32", {
                "alias": "demo", "head_index": 254,
                "kernel_id": "mth.u32.add_checked", "a": 7, "b": 9})
            call("nanlm_universe_renew", {"alias": "demo"})
            call("nanlm_universe_release", {"alias": "demo"})

        call("nanlm_status")
        call("nanlm_chart_storage", {"action": "inspect"})

        # Chat and transcripts.
        call("nanlm_tree_inspect", {})
        call("nanlm_tree_mutate", {
            "head": 2, "branch": "vector123", "leaf": "coordinate",
            "data_hex": "010203", "expected_revision": 0})
        call("nanlm_tree_inspect", {
            "head": 2, "branch": "vector123", "leaf": "coordinate"})
        call("nanlm_tree_audit", {})
        call("nanlm_chat_train", {
            "message": "Explore vector123 and propose a math experiment",
            "session_id": "demo-training"})
        call("nanlm_chat", {"message": "Hello from the demo", "session_id": "demo"})
        call("nanlm_chat_history", {"session_id": "demo"})
        call("nanlm_phone_call", {
            "action": "start", "session_id": "demo-phone",
            "message": "A river is water.", "addressee": "hermes"})
        call("nanlm_phone_call", {
            "action": "say", "session_id": "demo-phone",
            "message": "What is a river?", "addressee": "nanlm"})

        # Corpus generation and curriculum generation.
        _, corpus = call("nanlm_generate_corpus", {"topic": "river valley", "records": 64})
        call("nanlm_generate_curriculum", {"topic": "logic and language", "records_per_stage": 3})
        if corpus:
            corpus_path = corpus["corpus_path"]
        else:
            corpus_path = str(root / "demo-corpus.txt")
            Path(corpus_path).write_text(
                "The river flows through the valley.\nBirds fly above the water.\n" * 8,
                encoding="utf-8")
        call("nanlm_train", {"corpus_path": corpus_path, "duration_minutes": 1})
        call("nanlm_evaluate", {"corpus_path": corpus_path})
        call("nanlm_math_train", {"corpus_path": corpus_path, "epochs": 1, "max_samples_per_head": 16})

        # Structured training and the four explicit task presets.
        resolutions = ["bit", "nibble", "byte", "phonetic", "morpheme",
                       "lexical", "word_completion", "sentence", "paragraph", "narrative"]
        relation_records = []
        for i in range(4):
            relation_records.append({
                "split": "train", "source_digest": hashlib.sha256(str(i).encode()).hexdigest(),
                "item": {"kind": "relation", "subject": "demo-%d" % i, "predicate": "supports",
                         "object": "The river flows.", "polarity": "supports",
                         "confidence": 0.9, "evidence": {"demo": True},
                         "provenance": {"resolution": resolutions[i]}},
            })
        relation_records.append({
            "split": "test", "source_digest": "f" * 64,
            "item": {"kind": "relation", "subject": "demo-test", "predicate": "refutes",
                     "object": "The river is dry.", "polarity": "refutes",
                     "confidence": 0.8, "evidence": {"demo": True},
                     "provenance": {"resolution": "lexical"}},
        })
        call("nanlm_structured_relation_train", {"records": relation_records, "epochs": 1})
        call("nanlm_curriculum_experiment", {"topic": "logic", "records_per_stage": 3, "epochs": 1, "max_samples_per_head": 16})
        call("nanlm_logic_session", {"topic": "logic", "records_per_stage": 3, "epochs": 1, "max_samples_per_head": 16})
        call("nanlm_word_session", {"topic": "words", "records_per_stage": 3, "epochs": 1, "max_samples_per_head": 16})
        call("nanlm_sentence_session", {"topic": "sentences", "records_per_stage": 3, "epochs": 1, "max_samples_per_head": 16})
        call("nanlm_training_pipeline", {"topic": "pipeline", "records_per_stage": 3, "epochs": 1, "max_samples_per_head": 16})

        # Mathematical memory.
        discoveries = [{
            "kind": "relation", "subject": "demo-subject", "predicate": "supports",
            "object": "The river flows.", "resolution": 5, "polarity": "supports",
            "confidence": 0.9, "evidence": {"demo": True},
        }]
        call("nanlm_memory_ingest", {"discoveries": discoveries, "source_digest": "a" * 64, "head": 0, "layer": 5})
        call("nanlm_memory_reorganize", {})
        call("nanlm_memory_cross_resolution", {"limit": 100})
        call("nanlm_memory_merge", {"records": [{"item": discoveries[0], "source_digest": "b" * 64, "head": 0, "layer": 5}]})
        call("nanlm_memory_query", {"subject": "demo-subject", "limit": 20})
        call("nanlm_memory_assessments", {"subject": "demo-subject", "limit": 20})

        # Layered L0-L9 learning, 1 GiB paged scratch and sparse HW4 WordNet.
        call("nanlm_learning_cycle", {
            "message": "Copper is a metal because it conducts electricity.",
            "source": "demo",
            "relations": [{"subject": "copper", "predicate": "is", "object": "metal"}],
        })
        call("nanlm_learning_status")
        call("nanlm_learning_sync", {"approved": True})
        call("nanlm_wordnet_ingest", {"word": "copper", "depth": 2})
        call("nanlm_scratch_register", {"layer": 5, "limit": 8})

        # Resident mirror topology, Boolean-gated self-modification and MTH facade.
        call("nanlm_mirror_topology")
        call("nanlm_mirror_topology", {
            "route": "nan://GM0:0::SC0:0::MC0:0::BS0:0::HW0:0::HW0:7::L5"})
        call("nanlm_self_modify", {
            "action": "relation",
            "relation": {"subject": "copper", "predicate": "is", "object": "metal"},
            "layer": 5, "source_head": 1, "evidence_sha256": "a" * 64,
            "strength_q16": 32768, "polarity": 1, "expected_generation": 0})
        call("nanlm_self_modify", {"action": "status"})
        call("nanlm_mth_tools", {"tool": "mth.catalog"})
        call("nanlm_mth_tools", {"tool": "mth.inspect", "arguments": {"layer": 5}})
        call("nanlm_mth_tools", {
            "tool": "mth.route",
            "arguments": {"address": "nan://GM0:0::SC0:0::MC0:0::BS0:0::HW0:0::HW0:7::L5"}})
        teacher_records = [{
            "subject": "copper", "predicate": "is", "object": "metal",
            "definition": "a reddish metal", "layer": 5,
            "confidence": 0.9, "evidence": "demo teacher"}]
        with patch.object(gemma_teacher.LLMTeacher, "generate",
                          return_value=teacher_records):
            call("nanlm_teacher_teach", {"topic": "metals", "count": 2})

        # CLOCKED planning, execution, scratch, head process, synthesis.
        _, plan = call("nanlm_clocked_plan", {"question": "Check PII handling in the source", "sources": [
            {"id": "source-a", "text": "Contact demo@example.com for details."},
        ], "scratch_id": "demo-clocked"})
        call("nanlm_clocked_execute", {
            "question": "Check PII handling in the source",
            "sources": [{"id": "source-a", "text": "Contact demo@example.com for details."}],
            "scratch_id": "demo-clocked", "approved": True,
            "plan_sha256": plan["approval_digest"],
        })
        call("nanlm_clocked_execute", {
            "question": "Extract the supported claims",
            "sources": [{"id": "source-b", "text": "The river flows. Birds fly above the water."}],
            "scratch_id": "demo-clocked", "approved": True,
            "plan_sha256": call("nanlm_clocked_plan", {"question": "Extract the supported claims", "sources": [
                {"id": "source-b", "text": "The river flows. Birds fly above the water."},
            ], "scratch_id": "demo-clocked"})[1]["approval_digest"],
        })
        call("nanlm_clocked_scratch", {"scratch_id": "demo-clocked"})
        call("nanlm_head_process", {"question": "Validate the draft", "scratch_id": "demo-clocked",
                                     "purpose": "ANSWER_VALIDATION", "draft": "The river flows."})
        call("nanlm_clocked_synthesize", {"question": "What is supported?", "scratch_id": "demo-clocked", "max_tokens": 256})

        # Isolated candidate lifecycle.
        call("nanlm_head_candidate", {"mutation_id": "demo-candidate", "head_index": 42,
                                       "patch_mask": 1, "base_topology_digest": "a" * 64,
                                       "approved": True, "scratch_id": "demo-clocked"})
        call("nanlm_head_learn_candidate", {"mutation_id": "demo-learn", "base_topology_digest": "a" * 64,
                                             "approved": True, "scratch_id": "demo-clocked", "epochs": 1})
        call("nanlm_head_candidate_evaluate", {"mutation_id": "demo-learn", "records": [{
            "resolution": 5, "polarity": "+", "subject": "demo", "predicate": "supports",
            "object": "The river flows.", "source_digest": "c" * 64,
        }], "approved": True, "scratch_id": "demo-clocked"})

        # NaN16DE route uses a signed envelope; keep the demo local.
        def fake_send(endpoint, token, envelope, timeout=15):
            return {"accepted": True, "route": envelope.route.text(),
                    "route_bytes": envelope.route.bytes().hex(),
                    "word_raw": envelope.word.pack().hex(),
                    "opcode": envelope.word.opcode,
                    "installation_id": envelope.installation_id,
                    "body_sha256": envelope.as_dict()["body_sha256"]}
        import nanlmind_hermes.nan16de as nan16de
        with patch.object(nan16de, "send_envelope", side_effect=fake_send):
            call("nanlm_nan16de_route", {"route": "nan://HW2:BS0:MC0:SC0:GM0", "kind": "route"})

        call("nanlm_auth_revoke")

    authority_http.shutdown()
    authority_http.server_close()
    authority_thread.join(timeout=2)
    clocked_server.shutdown()
    clocked_server.server_close()
    clocked_thread.join(timeout=2)
    head_server.shutdown()
    head_server.server_close()
    head_thread.join(timeout=2)

    passed = sum(1 for row in results if row["status"] == "pass")
    report = {
        "schema": "nanlmind-hermes-demo/1",
        "version": "1.3.2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "native_binary": "nanlm-universal",
        "tools_total": len(ctx.tools),
        "tools_called": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    json_path = Path(args.json) if args.json else Path(args.output).with_name("demo-results.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Demo results",
        "",
        "Generated by `scripts/demo_run.py` against the real plugin registration,",
        "a loopback NANNET authority, local CLOCKED and head-cortex stubs, and the",
        "real `nanlm-universal` binary.",
        "",
        "| Tool | Status | Milliseconds | Result |",
        "|---|---:|---:|---|",
    ]
    for row in results:
        summary = row["summary"].replace("|", "\\|")
        lines.append("| `%s` | %s | %s | %s |" % (
            row["tool"], "PASS" if row["status"] == "pass" else "FAIL",
            row["ms"], summary))
    lines += [
        "",
        "**Summary:** %d/%d tools passed." % (passed, len(results)),
        "",
        "The JSON report is written to `%s`." % json_path.name,
        "",
    ]
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print("\n%s: %d/%d tools passed" % (args.output, passed, len(results)))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
