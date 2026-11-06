"""Single-writer, wall-clock-bounded native runner supervisor.

The native executable is an adapter boundary. It must accept:
  <runner argv> train --corpus PATH --duration-seconds N --checkpoint PATH
      --checkpoint-interval-seconds N
  <runner argv> evaluate --corpus PATH
and handle SIGINT by finishing its current bounded unit and atomically
publishing the checkpoint. The core repository's CLI is not assumed to expose
this adapter until that protocol is implemented there.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
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


def run_bounded(
    command: Sequence[str],
    corpus_path: Path,
    state_dir: Path,
    duration_minutes: int,
    checkpoint_interval_seconds: int = 60,
) -> dict:
    duration_minutes = validate_duration(duration_minutes)
    if not command or not all(isinstance(x, str) and x for x in command):
        raise RunnerError("runner_command is not configured as a non-empty argv list")
    if checkpoint_interval_seconds < 5 or checkpoint_interval_seconds > 3600:
        raise RunnerError("checkpoint interval must be between 5 and 3600 seconds")
    corpus_path = corpus_path.expanduser().resolve(strict=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "training.lock"
    status_path = state_dir / "last_run.json"
    checkpoint = state_dir / "checkpoint"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError("this universe already has an active training run") from exc
        digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
        budget = duration_minutes * 60
        started = time.monotonic()
        deadline = started + budget
        argv = list(command) + [
            "train",
            "--corpus", str(corpus_path),
            "--duration-seconds", str(budget),
            "--checkpoint", str(checkpoint),
            "--checkpoint-interval-seconds", str(checkpoint_interval_seconds),
        ]
        proc = subprocess.Popen(argv, cwd=str(state_dir))
        reason = "completed"
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            reason = "deadline"
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                reason = "deadline_forced_stop"
        elapsed = min(time.monotonic() - started, budget)
        result = {
            "status": "ok" if proc.returncode == 0 else "runner_exit",
            "exit_code": proc.returncode,
            "stop_reason": reason,
            "duration_budget_seconds": budget,
            "elapsed_seconds": round(elapsed, 3),
            "corpus_sha256": digest,
            "checkpoint_path": str(checkpoint) if checkpoint.exists() else None,
            "checkpoint_present": checkpoint.exists(),
        }
        _atomic_json(status_path, result)
        if proc.returncode not in (0, -signal.SIGINT, 130) and reason == "completed":
            raise RunnerError(f"native runner exited with code {proc.returncode}")
        return result
