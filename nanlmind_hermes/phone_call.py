"""Three-way phone-call chat: User ↔ Hermes ↔ NaNLM.

The call state machine is a raw NaN16 program, ``nan/phone_call.nan``,
executed on the NaNVM0 boundary (``nanvm.py``): validation, addressee
routing, round loops, event-plan emission and turn accounting all run as
raw 0xNaN words. This module only marshals text between the machine and
the participants (Hermes bridge, NaNLM native chat) and mirrors every turn
into the shared scratch journal, which is the third chat window. When the
call is started with ``clocked``, Hermes turns route through the CLOCKED
gateway and the plan/result/evidence lanes land in the same window.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .chat import chat as native_chat
from .nanvm import NanVM, make_word, parse_nan_image, word_payload
from .scratch import Scratch

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LANE = {"user": "phone_user", "hermes": "phone_hermes", "nanlm": "phone_nanlm"}
WRITER = {"user": "user", "hermes": "hermes", "nanlm": "nanlmind"}

D_BASE = 1024
A_ACTION, A_ADDRESSEE, A_ROUNDS, A_STATUS, A_TURN, A_HAS_MSG, A_COUNT, A_ERR = range(8)
EVENT_BASE = 1088

ACTIONS = {"start": 0, "say": 1, "run": 2, "status": 3, "end": 4}
ADDRESSEE_CODES = {"hermes": 0, "nanlm": 1, "both": 2}
ADDRESSEE_NAMES = {0: "hermes", 1: "nanlm", 2: "both", 3: "user+hermes"}
SPEAKERS = ("user", "hermes", "nanlm")
PHASES = ("user_message", "hermes_to_nanlm", "nanlm_reply",
          "hermes_followup", "hermes_to_user")
ERR_MESSAGES = {
    1: "phone call is closed",
    2: "rounds must be 1..64",
    3: "addressee must be hermes, nanlm or both",
}

_PROGRAM_CACHE = None


def _program():
    global _PROGRAM_CACHE
    if _PROGRAM_CACHE is None:
        path = Path(__file__).parent / "nan" / "phone_call.nan"
        _PROGRAM_CACHE = parse_nan_image(path.read_text(encoding="utf-8"))
    return _PROGRAM_CACHE


class PhoneCallError(ValueError):
    pass


class _NativeContext:
    """Force the NaNLM participant to use native parsing/synthesis."""

    def __init__(self, base):
        self._base = base

    def get_config(self, name, default=None):
        if name == "chat_mode":
            return "native"
        return self._base.get_config(name, default)


def _setting(ctx, key, default=None):
    try:
        return ctx.get_config(key, default=default)
    except Exception:
        return default


class PhoneCall:
    def __init__(self, ctx, session_id):
        if not SESSION_RE.fullmatch(session_id or ""):
            raise PhoneCallError("session_id must be 1..64 letters, digits, underscores or hyphens")
        self.ctx = ctx
        self.session_id = session_id
        root = Path(str(_setting(ctx, "state_dir", "~/.local/share/nanlmind-hermes"))).expanduser()
        self.root = root / "phone_calls"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass
        self.state_path = self.root / (session_id + ".json")
        # The call window lives in the shared clocked_scratch root so CLOCKED
        # tools can read the same journal by scratch_id "call-<session>".
        scratch_root = root / "clocked_scratch"
        self.scratch = Scratch(scratch_root, "call-" + session_id)
        self.state = self._load()

    def _load(self):
        if self.state_path.exists():
            try:
                value = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PhoneCallError("phone-call state is unreadable") from exc
            if value.get("schema") != "nanlm.phone-call/1":
                raise PhoneCallError("unsupported phone-call state")
            return value
        return {
            "schema": "nanlm.phone-call/1", "session_id": self.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": "open", "turn": 0, "history": [], "clocked": False,
        }

    def _save(self):
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state["history"] = self.state["history"][-500:]
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            temp.chmod(0o600)
        except OSError:
            pass
        temp.replace(self.state_path)

    def _append(self, speaker, text, phase, addressee, turn):
        text = str(text or "").strip()
        if not text:
            return None
        row = {
            "turn": turn, "speaker": speaker, "text": text[:8000],
            "phase": phase, "addressee": addressee,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        self.state["history"].append(row)
        self.state["turn"] = max(int(self.state.get("turn", 0)), turn)
        self.scratch.append(LANE[speaker], WRITER[speaker], row)
        self._save()
        return row

    def _kernel(self, action, addressee_code, rounds, has_msg):
        """Run the raw NaN16 routing kernel and decode its outputs."""
        vm = NanVM(_program())
        base = D_BASE
        for offset, value in (
            (A_ACTION, action),
            (A_ADDRESSEE, addressee_code),
            (A_ROUNDS, rounds),
            (A_STATUS, 0 if self.state.get("status") == "open" else 1),
            (A_TURN, int(self.state.get("turn", 0))),
            (A_HAS_MSG, 1 if has_msg else 0),
        ):
            vm.mem[base + offset] = make_word(value, 10)
        vm.run()
        err = word_payload(vm.mem[base + A_ERR])
        count = word_payload(vm.mem[base + A_COUNT])
        status = word_payload(vm.mem[base + A_STATUS])
        turn = word_payload(vm.mem[base + A_TURN])
        events = []
        for index in range(count):
            words = vm.mem[EVENT_BASE + 4 * index:EVENT_BASE + 4 * index + 4]
            events.append({
                "speaker": SPEAKERS[word_payload(words[0])],
                "phase": PHASES[word_payload(words[1])],
                "addressee": ADDRESSEE_NAMES.get(word_payload(words[2]), "hermes"),
            })
        return err, events, status, turn

    def _llm(self):
        return getattr(self.ctx, "llm", None)

    def _hermes_message(self, instruction: str, transcript: list[dict], fallback: str) -> str:
        llm = self._llm()
        if llm is None or not callable(getattr(llm, "complete", None)):
            return fallback
        system = (
            "You are Hermes, the language bridge in a three-way call with the User and NaNLM. "
            "NaNLM is a native relation learner, not a language model. Never answer factual "
            "questions on NaNLM's behalf and never claim NaNLM learned something it did not. "
            "Your job is to relay and, when the user addresses you directly, speak briefly. "
            "Keep every message to one or two short sentences. Output only the message."
        )
        messages = [{"role": "system", "content": system}]
        for row in transcript[-12:]:
            role = "user" if row["speaker"] in {"user", "nanlm"} else "assistant"
            prefix = {"user": "User", "nanlm": "NaNLM", "hermes": "Hermes"}[row["speaker"]]
            messages.append({"role": role, "content": f"[{prefix}] {row['text']}"})
        messages.append({"role": "user", "content": instruction})
        try:
            result = llm.complete(messages=messages, temperature=0.2, max_tokens=160,
                                  timeout=60, purpose="nanlm-phone-hermes")
            text = getattr(result, "text", "")
            return text.strip() if isinstance(text, str) and text.strip() else fallback
        except Exception:
            return fallback

    def _clocked_evidence(self) -> tuple[str, list[dict]]:
        """Run the pending question through CLOCKED; return (summary, atoms)."""
        from .clocked_client import ClockedGatewayClient
        from .clocked_mode import execute_multi_source
        from .hermes_secrets import get_secret

        url = str(_setting(self.ctx, "clocked_gateway_url", "") or "")
        token = get_secret("CLOCKED_GATEWAY_TOKEN")
        if not url or not token:
            return "", []
        history = [row for row in self.state["history"] if row["speaker"] == "user"]
        question = history[-1]["text"] if history else ""
        if not question:
            return "", []
        # One aggregated call-window source keeps the CLOCKED plan single-source
        # (extract -> grounding -> decide); the gateway does not accept a
        # cross-source compare plan for this task shape.
        window = "\n".join(
            "[%s] %s" % (row["speaker"], row["text"])
            for row in self.state["history"][-6:])
        sources = [{"id": "call-window", "text": window}]
        before = len(self.scratch.lane("evidence_atoms"))
        client = ClockedGatewayClient(base_url=url, token=token)
        execute_multi_source(client, self.scratch, question, sources)
        atoms = []
        for row in self.scratch.lane("evidence_atoms")[before:]:
            payload = row.get("payload", {})
            if isinstance(payload.get("object"), str) and payload.get("predicate") == "states_claim":
                atoms.append(payload["object"])
        summary = "CLOCKED evidence for this turn: " + "; ".join(atoms[:4]) if atoms else ""
        return summary, atoms

    def _hermes_turn(self, instruction: str, transcript: list[dict], fallback: str,
                     use_clocked: bool) -> str:
        prompt = instruction
        if use_clocked:
            try:
                summary, _atoms = self._clocked_evidence()
                if summary:
                    prompt = instruction + "\n" + summary
            except Exception:
                pass
        return self._hermes_message(prompt, transcript, fallback)

    def _nanlm_reply(self, message: str) -> dict:
        return native_chat(_NativeContext(self.ctx), message,
                           session_id="phone-" + self.session_id,
                           temperature=0.0, max_tokens=512)

    def _cycle(self, message=None, addressee="hermes", rounds=1, action="say",
               use_clocked=False):
        if isinstance(rounds, bool) or not isinstance(rounds, int):
            raise PhoneCallError("rounds must be 1..64")
        if addressee not in ADDRESSEE_CODES:
            raise PhoneCallError("addressee must be hermes, nanlm or both")
        text = str(message).strip() if message is not None else ""
        has_msg = bool(self.state["history"]) if action == "run" else bool(text)
        if action in ("start", "say") and not has_msg:
            raise PhoneCallError("message is required")

        err, events, status, turn = self._kernel(
            ACTIONS[action], ADDRESSEE_CODES[addressee], rounds, has_msg)
        if err:
            if err == 4:
                raise PhoneCallError(
                    "start the call with a user message" if action == "run"
                    else "message is required")
            raise PhoneCallError(ERR_MESSAGES.get(err, "phone call kernel error %d" % err))
        turn -= len(events)  # the kernel already added this plan to TURN

        history = list(self.state["history"])
        last_user = next((row["text"] for row in reversed(history)
                          if row["speaker"] == "user"), "")
        last_nanlm = ""
        pending = ""
        for event in events:
            turn += 1
            speaker, phase = event["speaker"], event["phase"]
            if speaker == "user":
                row_text = text
                last_user = text
                pending = text
            elif speaker == "hermes":
                if phase == "hermes_to_nanlm":
                    row_text = self._hermes_message(
                        "Forward the user's latest message to NaNLM. Preserve its content. "
                        "Output only the message to NaNLM.", self.state["history"],
                        fallback=last_user or "NaNLM, the user is waiting.")
                    pending = row_text
                elif phase == "hermes_followup":
                    row_text = self._hermes_turn(
                        "NaNLM just replied. Ask NaNLM one short follow-up question that "
                        "helps it clarify or extend the same topic. Output only the question.",
                        self.state["history"],
                        fallback="NaNLM, can you extend that answer with one concrete relation?",
                        use_clocked=use_clocked)
                    pending = row_text
                else:
                    row_text = self._hermes_turn(
                        "Relay NaNLM's latest reply to the user in one short sentence, without "
                        "adding facts.", self.state["history"],
                        fallback="Hermes: " + last_nanlm.strip(),
                        use_clocked=use_clocked)
            else:
                reply = self._nanlm_reply(pending or last_user)
                row_text = str(reply.get("response", ""))
                last_nanlm = row_text
            self._append(speaker, row_text, phase, event["addressee"], turn)

        if status == 1 and self.state.get("status") != "closed":
            self.state["status"] = "closed"
            self._save()
        if events:
            self.scratch.append("phone_event", "hermes", {
                "kind": "nan16_call_plan", "kernel": "nan16/1",
                "action": action, "addressee": addressee, "rounds": rounds,
                "events": [{"speaker": e["speaker"], "phase": e["phase"],
                            "addressee": e["addressee"]} for e in events],
                "turn": turn,
            })
        return self.snapshot()

    def start(self, message, addressee="hermes", rounds=1, clocked=False):
        self.state["clocked"] = bool(clocked)
        self._save()
        return self._cycle(message, addressee, rounds, action="start",
                           use_clocked=bool(clocked))

    def say(self, message, addressee="hermes", rounds=1, clocked=None):
        if clocked is not None:
            self.state["clocked"] = bool(clocked)
            self._save()
        return self._cycle(message, addressee, rounds, action="say",
                           use_clocked=bool(self.state.get("clocked")))

    def run(self, rounds=1):
        return self._cycle(None, "hermes", rounds, action="run",
                           use_clocked=bool(self.state.get("clocked")))

    def close(self):
        err, events, status, turn = self._kernel(ACTIONS["end"], 0, 1, False)
        self.state["status"] = "closed"
        self._save()
        return self.snapshot()

    def snapshot(self, limit=80):
        rows = self.state["history"][-limit:]
        return {
            "schema": "nanlm.phone-call/1", "session_id": self.session_id,
            "status": self.state.get("status"), "turn": self.state.get("turn"),
            "last_speaker": rows[-1]["speaker"] if rows else None,
            "history": rows,
            "scratch_id": self.scratch.scratch_id,
            "participants": ["user", "hermes", "nanlm"],
            "default_path": "user -> hermes -> nanlm -> user + hermes",
            "clocked": bool(self.state.get("clocked")),
            "kernel": "nan16/1",
        }


def handle(ctx, params):
    action = str(params.get("action", "say")).lower()
    session_id = str(params.get("session_id", "default"))
    call = PhoneCall(ctx, session_id)
    if action == "status":
        return call.snapshot()
    if action == "end":
        return call.close()
    rounds = params.get("rounds", 1)
    addressee = str(params.get("addressee", "hermes")).lower()
    message = params.get("message")
    clocked = params.get("clocked")
    if action == "start":
        return call.start(message or "Hello.", addressee, rounds, clocked=bool(clocked))
    if action == "say":
        if not message:
            raise PhoneCallError("say requires a message")
        return call.say(message, addressee, rounds,
                        clocked=None if clocked is None else bool(clocked))
    if action == "run":
        return call.run(rounds)
    raise PhoneCallError("action must be start, say, run, status or end")
