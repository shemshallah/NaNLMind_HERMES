import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes import phone_call
from nanlmind_hermes.scratch import Scratch


class Context:
    def __init__(self, root):
        self.config = {"state_dir": str(root), "chat_mode": "native", "chat_learning": False}

    def get_config(self, key, default=None):
        return self.config.get(key, default)


class PhoneCallTests(unittest.TestCase):
    def test_default_cycle_user_hermes_nanlm_user_and_hermes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(Path(tmp))
            result = phone_call.handle(ctx, {
                "action": "start", "session_id": "call-1",
                "message": "A river is water.", "addressee": "hermes",
            })
            speakers = [row["speaker"] for row in result["history"]]
            self.assertEqual(speakers, ["user", "hermes", "nanlm", "hermes"])
            nanlm = next(row for row in result["history"] if row["speaker"] == "nanlm")
            self.assertIn("river", nanlm["text"].lower())
            self.assertTrue(nanlm["text"].endswith((".", "!", "?")))
            self.assertTrue(result["history"][-1]["text"].startswith("Hermes:"))
            self.assertEqual(result["default_path"], "user -> hermes -> nanlm -> user + hermes")
            self.assertEqual(result["kernel"], "nan16/1")

    def test_user_can_address_nanlm_directly_and_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(Path(tmp))
            phone_call.handle(ctx, {"action": "start", "session_id": "call-2",
                                    "message": "A river is water."})
            result = phone_call.handle(ctx, {
                "action": "say", "session_id": "call-2",
                "message": "What is a river?", "addressee": "nanlm",
            })
            self.assertEqual(result["history"][-3]["speaker"], "user")
            self.assertEqual(result["history"][-2]["speaker"], "nanlm")
            self.assertEqual(result["history"][-1]["speaker"], "hermes")
            self.assertIn("river", result["history"][-3]["text"].lower())
            self.assertIn("river", result["history"][-2]["text"].lower())

    def test_arbitrary_rounds_and_shared_scratch_lanes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(Path(tmp))
            phone_call.handle(ctx, {"action": "start", "session_id": "call-3",
                                    "message": "A river is water."})
            result = phone_call.handle(ctx, {"action": "run", "session_id": "call-3", "rounds": 3})
            self.assertGreaterEqual(sum(1 for row in result["history"] if row["speaker"] == "nanlm"), 4)
            self.assertGreaterEqual(sum(1 for row in result["history"] if row["speaker"] == "hermes"), 5)
            scratch = Scratch(Path(tmp) / "clocked_scratch", "call-call-3")
            lanes = {row["lane"] for row in scratch.snapshot()["entries"]}
            self.assertTrue({"phone_user", "phone_hermes", "phone_nanlm", "phone_event"} <= lanes)

    def test_state_persists_between_calls_and_can_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(Path(tmp))
            phone_call.handle(ctx, {"action": "start", "session_id": "call-4",
                                    "message": "Birds can fly."})
            status = phone_call.handle(ctx, {"action": "status", "session_id": "call-4"})
            self.assertEqual(status["turn"], 4)
            closed = phone_call.handle(ctx, {"action": "end", "session_id": "call-4"})
            self.assertEqual(closed["status"], "closed")
            with self.assertRaises(phone_call.PhoneCallError):
                phone_call.handle(ctx, {"action": "say", "session_id": "call-4",
                                        "message": "Hello again."})

    def test_kernel_event_plan_matches_routing_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(Path(tmp))
            call = phone_call.PhoneCall(ctx, "kernel-1")
            err, events, status, turn = call._kernel(
                phone_call.ACTIONS["say"], phone_call.ADDRESSEE_CODES["hermes"], 2, True)
            self.assertEqual(err, 0)
            self.assertEqual(
                [(e["speaker"], e["phase"]) for e in events],
                [("user", "user_message"), ("hermes", "hermes_to_nanlm"),
                 ("nanlm", "nanlm_reply"), ("hermes", "hermes_followup"),
                 ("nanlm", "nanlm_reply"), ("hermes", "hermes_to_user")])
            self.assertEqual(status, 0)
            self.assertEqual(turn, 6)
            self.assertEqual(events[0]["addressee"], "hermes")
            self.assertEqual(events[2]["addressee"], "user+hermes")

            err, events, _status, _turn = call._kernel(
                phone_call.ACTIONS["say"], phone_call.ADDRESSEE_CODES["nanlm"], 1, True)
            self.assertEqual(err, 0)
            self.assertNotIn("hermes_to_nanlm", [e["phase"] for e in events])
            self.assertEqual(events[0]["addressee"], "nanlm")

            err, events, _status, _turn = call._kernel(
                phone_call.ACTIONS["say"], phone_call.ADDRESSEE_CODES["both"], 1, True)
            self.assertEqual(events[0]["addressee"], "both")

    def test_kernel_rejects_closed_calls_bad_rounds_addressee_and_missing_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(Path(tmp))
            call = phone_call.PhoneCall(ctx, "kernel-2")
            err, _e, _s, _t = call._kernel(
                phone_call.ACTIONS["say"], 0, 0, True)
            self.assertEqual(err, 2)
            err, _e, _s, _t = call._kernel(
                phone_call.ACTIONS["say"], 0, 65, True)
            self.assertEqual(err, 2)
            err, _e, _s, _t = call._kernel(
                phone_call.ACTIONS["say"], 3, 1, True)
            self.assertEqual(err, 3)
            err, _e, _s, _t = call._kernel(
                phone_call.ACTIONS["run"], 0, 1, False)
            self.assertEqual(err, 4)
            call.state["status"] = "closed"
            err, events, status, _t = call._kernel(
                phone_call.ACTIONS["say"], 0, 1, True)
            self.assertEqual(err, 1)
            self.assertEqual(events, [])
            err, events, status, _t = call._kernel(
                phone_call.ACTIONS["status"], 0, 1, False)
            self.assertEqual(err, 0)
            self.assertEqual(events, [])
            err, events, status, _t = call._kernel(
                phone_call.ACTIONS["end"], 0, 1, False)
            self.assertEqual(err, 0)
            self.assertEqual(status, 1)

    def test_plan_recorded_in_phone_event_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(Path(tmp))
            phone_call.handle(ctx, {"action": "start", "session_id": "call-5",
                                    "message": "A river is water."})
            scratch = Scratch(Path(tmp) / "clocked_scratch", "call-call-5")
            events = [row["payload"] for row in scratch.lane("phone_event")]
            self.assertTrue(events)
            self.assertEqual(events[-1]["kernel"], "nan16/1")
            self.assertEqual(events[-1]["kind"], "nan16_call_plan")


if __name__ == "__main__":
    unittest.main()
