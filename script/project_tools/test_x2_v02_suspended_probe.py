import subprocess
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import x2_v02_suspended_probe as probe


class ProbeTests(unittest.TestCase):
    def test_ros_cleanup_is_in_finally_after_recovery(self):
        source = inspect.getsource(probe.main)
        recovery = source.index('emit("official_restored"')
        cleanup = source.index('finally:', recovery)
        self.assertGreater(source.index('node.destroy_node()', cleanup), cleanup)
        self.assertIn('finally:\n                    rclpy.shutdown()', source)

    def test_suspended_window_is_bounded_even_after_native_safety_trip(self):
        self.assertEqual(probe.POWERED_WINDOW_S, 3.0)
        source = inspect.getsource(probe.main)
        self.assertIn('started + (POWERED_WINDOW_S if powered else 35)', source)
        self.assertIn('emit("powered_window_complete"', source)

    def test_mc_discovery_settle_does_not_change_native_feedback_threshold(self):
        self.assertEqual(probe.MC_DISCOVERY_SETTLE_S, 3.0)
        source = inspect.getsource(probe.main)
        self.assertIn('now - stop_reply >= MC_DISCOVERY_SETTLE_S and fresh()', source)
        self.assertNotIn('--state-stale', source)

    def test_no_additional_runtime_freshness_abort(self):
        source = inspect.getsource(probe.main)
        self.assertNotIn('powered and not fresh()', source)
        self.assertNotIn('<= 0.1', source)
        self.assertIn('<= 0.5', source)

    def test_observe_never_enables_writer(self):
        args = probe.arguments(Path("/candidate"), Path("/isolated"), False)
        self.assertIn("--no-publish", args)
        self.assertNotIn("--dry-run", args)
        self.assertNotIn("--start-trigger-sentinel", args)
        self.assertEqual(args[args.index("--max-duration")+1], "10")

    def test_powered_stays_gated_until_unique_trigger(self):
        args = probe.arguments(Path("/candidate"), Path("/isolated"), True)
        self.assertEqual(args[args.index("--max-duration")+1], "3")
        self.assertEqual(args[args.index("--start-trigger-sentinel")+1],
                         "/isolated/start.trigger")
        self.assertNotIn("--no-publish", args)
        self.assertEqual(args[args.index("--hold-for-mc-timeout-s")+1], "0")

    def test_modes_only_change_duration_and_writer_gate(self):
        def common(args):
            args = list(args)
            index = args.index("--max-duration")
            del args[index:index+2]
            for flag in ("--start-trigger-sentinel", "--ready-sentinel"):
                if flag in args:
                    index = args.index(flag)
                    del args[index:index+2]
            return [a for a in args if a != "--no-publish"]
        self.assertEqual(common(probe.arguments(Path("/c"), Path("/i"), True)),
                         common(probe.arguments(Path("/c"), Path("/i"), False)))

    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture"
            path.write_bytes(b"fixture")
            with self.assertRaises(RuntimeError):
                probe.check_hash(path, "0"*64)

    @patch.object(probe.os, "killpg")
    def test_already_exited_child_not_signalled(self, killpg):
        child = Mock()
        child.poll.return_value = 0
        probe.stop_child(child)
        killpg.assert_not_called()

    @patch.object(probe.os, "killpg")
    def test_termination_targets_only_own_session(self, killpg):
        child = Mock(pid=12345)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("fixture", 2), 0]
        probe.stop_child(child)
        self.assertEqual([call.args[0] for call in killpg.call_args_list], [12345, 12345])


if __name__ == "__main__":
    unittest.main()
