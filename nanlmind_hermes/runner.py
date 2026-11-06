"""Local NaNLMind native training/evaluation harness.

The native CLI's supported training entry point is `bootstrap CORPUS EPOCHS CHUNKS`.
This adapter invokes that real implementation locally, records its complete
stdout/stderr and reproducibility manifest, and runs independent byte/Kneser-Ney
controls. It deliberately does not label a log/manifest as model weights:
the current native Context2Model bootstrap is in-memory and is not resumable.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Sequence

from .policy import validate_duration


class RunnerError(RuntimeError):
    pass


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _command(command: Sequence[str]) -> list[str]:
    if isinstance(command, (str, bytes)) or not command or not all(isinstance(x, str) and x for x in command):
        raise RunnerError("runner_command must be a non-empty argv list pointing to the local nanlm-universal executable")
    return list(command)


def _run(argv: list[str], timeout: float) -> dict:
    started = time.monotonic()
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RunnerError(f"cannot start local native runner: {exc}") from exc
    try:
        output, _ = proc.communicate(timeout=max(0.1, timeout))
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.send_signal(signal.SIGINT)
        try:
            output, _ = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                output, _ = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                output, _ = proc.communicate()
    return {"argv": argv, "exit_code": proc.returncode, "timed_out": timed_out,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "output": output or ""}


def _metrics(output: str) -> dict:
    metrics = {}
    for key in ("bpc", "top1", "alpha_top1", "word_start_top1", "heldout_accuracy",
                "majority_baseline_accuracy", "accuracy", "acc"):
        matches = re.findall(r"\b" + key + r"=([0-9]+(?:\.[0-9]+)?)", output)
        if matches:
            try:
                metrics[key] = float(matches[-1])
            except ValueError:
                pass
    return metrics


def run_bounded(command: Sequence[str], corpus_path: Path, state_dir: Path,
                duration_minutes: int, checkpoint_interval_seconds: int = 60) -> dict:
    duration_minutes = validate_duration(duration_minutes)
    if isinstance(checkpoint_interval_seconds, bool) or not isinstance(checkpoint_interval_seconds, int) or not 5 <= checkpoint_interval_seconds <= 3600:
        raise RunnerError("checkpoint interval must be an integer between 5 and 3600 seconds")
    base = _command(command)
    corpus_path = corpus_path.expanduser().resolve(strict=True)
    corpus_digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    state_dir.mkdir(parents=True, exist_ok=True)
    artifacts = state_dir / "runs"
    artifacts.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    run_dir = artifacts / run_id
    run_dir.mkdir()
    status_path = state_dir / "last_run.json"
    lock_path = state_dir / "training.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError("this local universe already has an active training run") from exc

        budget = duration_minutes * 60
        # The native bootstrap CLI accepts epochs and chunk count. Epochs are
        # capped by the requested wall budget; the monotonic timeout is final.
        epochs = max(1, min(360, duration_minutes))
        train = _run(base + ["bootstrap", str(corpus_path), str(epochs), "8"], budget)
        log_path = run_dir / "training.log"
        log_path.write_text(train["output"], encoding="utf-8")
        result = {
            "schema": "nanlm-hermes-run/2", "run_id": run_id,
            "status": "ok" if train["exit_code"] == 0 and not train["timed_out"] else ("deadline" if train["timed_out"] else "runner_exit"),
            "exit_code": train["exit_code"], "stop_reason": "deadline" if train["timed_out"] else "completed",
            "duration_budget_seconds": budget, "training_elapsed_seconds": train["elapsed_seconds"],
            "corpus_path": str(corpus_path), "corpus_sha256": corpus_digest,
            "native_command": base, "native_subcommand": "bootstrap",
            "epochs_requested": epochs, "chunks": 8, "training_metrics": _metrics(train["output"]),
            "training_log": str(log_path), "artifact_dir": str(run_dir),
            "checkpoint_present": False, "resume_supported": False,
            "checkpoint_note": "Native Context2Model bootstrap is currently in-memory; this artifact is a reproducibility record, not model weights.",
        }
        _atomic_json(run_dir / "manifest.json", result)
        _atomic_json(status_path, result)
        if train["exit_code"] not in (0, -signal.SIGINT, 130) and not train["timed_out"]:
            raise RunnerError(f"native bootstrap exited with code {train['exit_code']}; see {log_path}")
        return result


def evaluate_native(command: Sequence[str], corpus_path: Path, state_dir: Path,
                    timeout_seconds: int = 300) -> dict:
    base = _command(command)
    corpus_path = corpus_path.expanduser().resolve(strict=True)
    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    results = {}
    for name, args in (
        ("topology_byte", ["byte", str(corpus_path)]),
        ("kneser_ney_order2", ["kn", str(corpus_path), "2"]),
        ("context2", ["ctx2", str(corpus_path)]),
    ):
        item = _run(base + args, timeout_seconds)
        results[name] = {k: item[k] for k in ("exit_code", "timed_out", "elapsed_seconds")}
        metric_output = item["output"]
        if name == "topology_byte":
            # The byte CLI prints topology, reference and order-0 baselines.
            # Score the topology line only; taking the final bpc silently
            # reports the weaker order-0 baseline as the model's result.
            topology_lines = [line for line in metric_output.splitlines()
                              if line.lstrip().lower().startswith("topology ")]
            if topology_lines:
                metric_output = topology_lines[-1]
        results[name]["metrics"] = _metrics(metric_output)
        results[name]["output"] = item["output"][-12000:]
        if item["timed_out"]:
            raise RunnerError(f"{name} evaluation exceeded {timeout_seconds}s")
        if item["exit_code"] != 0:
            raise RunnerError(f"{name} evaluation failed; see returned output")
    state_dir.mkdir(parents=True, exist_ok=True)
    return {"corpus_sha256": digest, "evaluations": results,
            "comparison_note": "These are independent native CLI runs with the CLI's documented 80/20 split; compare BPC/top-1 on the same corpus, not across different splits."}


def run_logic_train(command: Sequence[str], state_path: Path,
                    boolean_reps: int = 32, circuit_reps: int = 64,
                    timeout_seconds: int = 300) -> dict:
    """Run the native full-Boolean foundation and persist its state.

    This is a prerequisite gate, not a language-quality metric. The native
    session trains every one of the 256 heads on all 16 two-input Boolean
    functions (4 truth-table rows each), then tests composed gates and the
    word-after-logic probes.
    """
    base = _command(command)
    state_path = state_path.expanduser().resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    result = _run(base + ["logic-train", str(max(1, int(boolean_reps))),
                          str(max(1, int(circuit_reps))), str(state_path)],
                  timeout_seconds)
    output = result["output"]
    if result["exit_code"] != 0 or result["timed_out"]:
        raise RunnerError("native logic-train failed: " + output[-4000:])
    if "result=PASS" not in output:
        raise RunnerError("native Boolean foundation did not pass: " + output[-4000:])
    stages = []
    for line in output.splitlines():
        if not line.startswith("stage="):
            continue
        match = re.search(r"stage=(\S+) passed=(yes|no) correct=(\d+)/(\d+) accuracy=([0-9.]+)", line)
        if not match:
            continue
        distinct = re.search(r"distinct_rules=(\d+)", line)
        stages.append({
            "name": match.group(1), "passed": match.group(2) == "yes",
            "correct": int(match.group(3)), "total": int(match.group(4)),
            "accuracy": float(match.group(5)),
            "distinct_rules": int(distinct.group(1)) if distinct else None,
        })
    boolean = next((s for s in stages if s["name"] == "boolean_foundation"), None)
    if not boolean or not boolean["passed"] or boolean["correct"] != boolean["total"]:
        raise RunnerError("native Boolean foundation is incomplete or not exact")
    if boolean["total"] < 1024:
        raise RunnerError("native Boolean foundation did not cover 256 heads x 4 truth rows")
    if boolean.get("distinct_rules") != 16:
        raise RunnerError("native Boolean foundation did not produce all 16 distinct learned rules")
    if not state_path.is_file() or state_path.stat().st_size == 0:
        raise RunnerError("native Boolean foundation did not persist its state")
    digest_match = re.search(r"state_digest=([0-9a-f]{64})", output)
    return {
        "schema": "nanlm-hermes-boolean-foundation/1",
        "status": "passed", "native_command": result["argv"],
        "elapsed_seconds": result["elapsed_seconds"],
        "state_path": str(state_path), "state_bytes": state_path.stat().st_size,
        "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "state_digest": digest_match.group(1) if digest_match else None,
        "stages": stages, "stdout": output[-12000:],
    }


def run_math_train(command: Sequence[str], corpus_path: Path, state_dir: Path,
                   epochs: int = 3, max_samples_per_head: int = 4096) -> dict:
    """Train all native heads, resume the serialized global object, and emit typed discoveries."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100:
        raise RunnerError("math-train epochs must be between 1 and 100")
    if isinstance(max_samples_per_head, bool) or not isinstance(max_samples_per_head, int) or not 1 <= max_samples_per_head <= 100000:
        raise RunnerError("max_samples_per_head must be between 1 and 100000")
    base = _command(command)
    corpus_path = corpus_path.expanduser().resolve(strict=True)
    corpus_digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    state_dir.mkdir(parents=True, exist_ok=True)
    native_dir = state_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    traces = state_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    state_path = native_dir / "global_universe.nmg"
    trace_path = traces / (run_id + ".jsonl")
    lock_path = state_dir / "native_math_train.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError("native mathematical universe already has an active training run") from exc
        result = _run(base + ["math-train", str(corpus_path), str(epochs),
                              str(max_samples_per_head), str(state_path), str(trace_path)],
                      max(30.0, min(3600.0, epochs * 30.0)))
    if result["exit_code"] != 0 or result["timed_out"]:
        raise RunnerError("native math-train failed: " + result["output"][-4000:])
    if not state_path.is_file() or state_path.stat().st_size == 0:
        raise RunnerError("native math-train did not produce a serialized global universe")
    if not trace_path.is_file():
        raise RunnerError("native math-train did not produce a discovery trace")
    records = []
    seen_heads = set()
    for line_no, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunnerError(f"invalid native discovery JSON at line {line_no}") from exc
        if event.get("schema") != "nanlm-hermes-discovery/1":
            raise RunnerError(f"unsupported native discovery schema at line {line_no}")
        head = event.get("head")
        layer = event.get("layer", 0)
        if isinstance(head, bool) or not isinstance(head, int) or not 0 <= head < 256 or head in seen_heads:
            raise RunnerError(f"invalid or duplicate native head at line {line_no}")
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < 7:
            raise RunnerError(f"invalid native layer at line {line_no}")
        if event.get("source_digest") != corpus_digest:
            raise RunnerError(f"native discovery source digest mismatch at line {line_no}")
        hyperedge_id = event.get("hyperedge_id")
        hyperedge_label = event.get("hyperedge_label")
        if isinstance(hyperedge_id, bool) or not isinstance(hyperedge_id, int) or hyperedge_id <= 0:
            raise RunnerError(f"native discovery lacks a valid committed hyperedge at line {line_no}")
        if isinstance(hyperedge_label, bool) or not isinstance(hyperedge_label, int) or not 0 <= hyperedge_label < 2**64:
            raise RunnerError(f"native discovery has an invalid hyperedge label at line {line_no}")
        item = {key: event[key] for key in
                ("kind", "subject", "expression", "evidence", "provenance", "confidence", "polarity")
                if key in event}
        item.setdefault("provenance", {})
        item["provenance"]["native_address"] = event.get("address")
        item["provenance"]["native_head"] = head
        item["provenance"]["native_hyperedge_id"] = hyperedge_id
        item["provenance"]["native_hyperedge_label"] = hyperedge_label
        records.append({"item": item, "source_digest": corpus_digest,
                        "head": head, "layer": layer})
        seen_heads.add(head)
    if seen_heads != set(range(256)):
        raise RunnerError(f"native trace must include all 256 heads; received {len(seen_heads)}")
    return {
        "schema": "nanlm-hermes-native-math-run/1", "run_id": run_id,
        "status": "ok", "native_command": result["argv"],
        "elapsed_seconds": result["elapsed_seconds"], "corpus_sha256": corpus_digest,
        "native_state_path": str(state_path), "native_state_bytes": state_path.stat().st_size,
        "discovery_trace_path": str(trace_path), "discoveries": len(records),
        "head_count": len(seen_heads), "metrics": _metrics(result["output"]),
        "stdout": result["output"][-12000:], "records": records,
    }


def run_relation_train(command: Sequence[str], records_path: Path, state_dir: Path,
                       timeout_seconds: int = 300) -> dict:
    """Apply typed L0-L9 relation events to the same persistent native universe.

    N16REL/1 separates train and test rows. Only train rows mutate heads,
    canonical mirror-pair cells, temporal links and native hyperedges.
    """
    base = _command(command)
    records_path = records_path.expanduser().resolve(strict=True)
    raw = records_path.read_bytes()
    if not raw.startswith(b"N16REL/1\n"):
        raise RunnerError("relation corpus must use N16REL/1")
    corpus_digest = hashlib.sha256(raw).hexdigest()
    state_dir.mkdir(parents=True, exist_ok=True)
    native_dir = state_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    state_path = native_dir / "global_universe.nmg"
    lock_path = state_dir / "native_math_train.lock"
    before_sha = hashlib.sha256(state_path.read_bytes()).hexdigest() if state_path.is_file() else None
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError("native mathematical universe already has an active training run") from exc
        result = _run(base + ["relation-train", str(records_path), str(state_path)], timeout_seconds)
    if result["exit_code"] != 0 or result["timed_out"]:
        raise RunnerError("native relation-train failed: " + result["output"][-5000:])
    if not state_path.is_file() or state_path.stat().st_size == 0:
        raise RunnerError("native relation-train did not persist the global universe")
    summary = re.search(
        r"relation_train status=updated train_events=(\d+) duplicate_events=(\d+) "
        r"changed_heads=(\d+) mirror_pair_updates=(\d+) hyperedges_added=(\d+) "
        r"state_bytes=(\d+) state_digest_before=([0-9a-f]{64}) state_digest_after=([0-9a-f]{64})",
        result["output"])
    if not summary:
        raise RunnerError("native relation-train omitted its mutation receipt: " + result["output"][-3000:])
    train_events, duplicates, changed_heads, pair_updates, edges, state_bytes = map(
        int, summary.groups()[:6])
    digest_before, digest_after = summary.group(7), summary.group(8)
    if not train_events or not changed_heads or not pair_updates or not edges:
        raise RunnerError("native relation-train reported no effective head/mirror/hyperedge mutation")
    if digest_before == digest_after:
        raise RunnerError("native relation-train claimed updates but global state digest did not change")
    if state_bytes != state_path.stat().st_size:
        raise RunnerError("native relation-train state byte count does not match persisted file")
    resolution_metrics = {}
    for match in re.finditer(
        r"resolution_L(\d+) train=(\d+) test=(\d+) accuracy=([0-9.]+) "
        r"balanced_accuracy=([0-9.]+) majority_baseline=([0-9.]+)", result["output"]):
        resolution_metrics[int(match.group(1))] = {
            "train": int(match.group(2)), "test": int(match.group(3)),
            "relation_polarity_accuracy": float(match.group(4)),
            "balanced_accuracy": float(match.group(5)),
            "majority_baseline": float(match.group(6)),
        }
    after_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
    return {
        "schema": "nanlm-hermes-native-relation-run/1",
        "status": "updated", "native_command": result["argv"],
        "elapsed_seconds": result["elapsed_seconds"],
        "corpus_sha256": corpus_digest, "records_path": str(records_path),
        "native_state_path": str(state_path), "native_state_bytes": state_bytes,
        "native_state_sha256_before": before_sha, "native_state_sha256_after": after_sha,
        "state_digest_before": digest_before, "state_digest_after": digest_after,
        "train_events": train_events, "duplicate_events": duplicates,
        "changed_heads": changed_heads, "mirror_pair_updates": pair_updates,
        "hyperedges_added": edges, "resolution_metrics": resolution_metrics,
        "metric_scope": "binary relation-polarity prediction by declared resolution; not text-generation accuracy",
        "stdout": result["output"][-12000:],
    }
