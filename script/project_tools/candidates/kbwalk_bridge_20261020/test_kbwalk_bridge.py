#!/usr/bin/env python3
"""Unit tests for kbwalk_bridge (no robot, no GUI, no network bind)."""
import json, os, sys, tempfile, time, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbwalk_bridge import IntentState, encode_cmd, parse_script, Sender, run_script


class TestIntentState(unittest.TestCase):
    def test_deadman_required(self):
        st = IntentState()
        st.key("w", True)
        self.assertEqual(st.intent(), (0.0, 0.0, 0.0))  # no deadman -> zeros

    def test_deadman_plus_w(self):
        st = IntentState()
        st.key("space", True)
        st.key("w", True)
        self.assertEqual(st.intent(), (1.0, 0.0, 0.0))

    def test_keyup_stops(self):
        st = IntentState()
        st.key("space", True)
        st.key("w", True)
        st.key("w", False)
        self.assertEqual(st.intent(), (0.0, 0.0, 0.0))

    def test_deadman_release_zeros(self):
        st = IntentState()
        st.key("space", True)
        st.key("w", True)
        st.key("space", False)
        self.assertEqual(st.intent(), (0.0, 0.0, 0.0))

    def test_focus_loss_zeros(self):
        st = IntentState()
        st.key("space", True)
        st.key("w", True)
        st.focus(False)
        self.assertEqual(st.intent(), (0.0, 0.0, 0.0))

    def test_turn_and_clamp(self):
        st = IntentState()
        st.key("space", True)
        st.key("a", True)
        st.key("d", True)  # opposite keys cancel
        self.assertEqual(st.intent(), (0.0, 0.0, 0.0))

    def test_back_reduced(self):
        st = IntentState()
        st.key("space", True)
        st.key("s", True)
        fwd, _, _ = st.intent()
        self.assertEqual(fwd, -0.6)


class TestEncode(unittest.TestCase):
    def test_multipart_shape(self):
        parts = encode_cmd(0.6, 0.0, 0.1)
        self.assertEqual(parts[0], b"planner_cmd")
        d = json.loads(parts[1].decode("utf-8"))
        self.assertEqual(d["intent"], "locomotion")
        self.assertEqual(d["magnitude"], "continuous")
        self.assertAlmostEqual(d["stick_fwd"], 0.6)
        self.assertAlmostEqual(d["stick_side"], 0.0)
        self.assertAlmostEqual(d["stick_yaw"], 0.1)

    def test_zeros_are_valid_stop_request(self):
        parts = encode_cmd(0, 0, 0)
        d = json.loads(parts[1])
        self.assertEqual((d["stick_fwd"], d["stick_side"], d["stick_yaw"]), (0.0, 0.0, 0.0))


class TestScript(unittest.TestCase):
    def test_parse(self):
        steps = parse_script(["walk 2 0.4", "turn 1 0.3", "idle 1"])
        self.assertEqual(steps, [(2.0, 0.4, 0.0), (1.0, 0.0, 0.3), (1.0, 0.0, 0.0)])

    def test_parse_rejects_unknown(self):
        with self.assertRaises(ValueError):
            parse_script(["jump 1"])

    def test_run_script_sends_zeros_tail(self):
        with tempfile.TemporaryDirectory() as td:
            sender = Sender("unused", os.path.join(td, "log.jsonl"), dry=True)
            run_script(sender, [(0.1, 0.4, 0.0)])
            sender.close()
            lines = [json.loads(l) for l in open(os.path.join(td, "log.jsonl"))]
            walk = [l for l in lines if l["cmd"]["stick_fwd"] == 0.4]
            stop = [l for l in lines if l["note"] == "script-stop"]
            self.assertGreaterEqual(len(walk), 1)
            self.assertGreaterEqual(len(stop), 20)  # >=2s of 20Hz zeros
            self.assertEqual(set(stop[0]["cmd"][k] for k in ("stick_fwd", "stick_side", "stick_yaw")), {0.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
