from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import unittest

try:
    from .state_machine import (
        Command,
        Config,
        GroupedFreshness,
        Machine,
        Observation,
        OutputContinuity,
        State,
        StreamHealth,
        TargetMode,
        smoothstep5,
    )
except ImportError:  # Direct execution from this directory.
    from state_machine import (
        Command,
        Config,
        GroupedFreshness,
        Machine,
        Observation,
        OutputContinuity,
        State,
        StreamHealth,
        TargetMode,
        smoothstep5,
    )


def healthy(stable: float = 5.0) -> StreamHealth:
    return StreamHealth(True, True, 0.002, 0.003, 0.001, 0.004, stable)


def continuity() -> OutputContinuity:
    return OutputContinuity(True, True, 0.001, 0.003, True)


def observation(**changes) -> Observation:
    base = Observation(
        system_state="Business",
        system_status_ready=True,
        feedback=healthy(),
        official=healthy(),
        sonic=healthy(),
        output=continuity(),
        official_powered=True,
        official_stationary=True,
        official_contract_complete=True,
        official_sample_after_handback=False,
        target_error_max_rad=0.01,
        measured_speed_max_rad_s=0.01,
        tracking_error_max_rad=0.01,
        vibration_velocity_rms_rad_s=0.01,
        vibration_accel_rms_rad_s2=0.5,
        vibration_reversal_hz=0.5,
    )
    return replace(base, **changes)


class FreshnessTests(unittest.TestCase):
    def test_uses_source_age_and_local_receive_age(self) -> None:
        tracker = GroupedFreshness(("leg", "arm"))
        tracker.observe("leg", 1, 10.000, 0.004)
        tracker.observe("arm", 1, 10.006, 0.030)
        health = tracker.health(10.020)
        self.assertAlmostEqual(health.local_age_s, 0.020)
        self.assertAlmostEqual(health.source_age_s, 0.044)
        self.assertAlmostEqual(health.group_skew_s, 0.006)

    def test_missing_group_is_incomplete(self) -> None:
        tracker = GroupedFreshness(("leg", "arm"))
        tracker.observe("leg", 1, 1.0, 0.0)
        self.assertFalse(tracker.health(1.01).complete)

    def test_replay_latches_sequence_fault(self) -> None:
        tracker = GroupedFreshness(("leg",))
        self.assertTrue(tracker.observe("leg", 10, 1.0, 0.0))
        self.assertFalse(tracker.observe("leg", 10, 1.1, 0.0))
        self.assertFalse(tracker.health(1.2).sequence_ok)

    def test_uint32_wrap_is_accepted(self) -> None:
        tracker = GroupedFreshness(("leg",))
        tracker.observe("leg", 0xFFFFFFFF, 1.0, 0.0)
        self.assertTrue(tracker.observe("leg", 0, 1.1, 0.0))

    def test_source_age_advances_between_callbacks(self) -> None:
        tracker = GroupedFreshness(("leg",))
        tracker.observe("leg", 1, 1.0, 0.02)
        self.assertAlmostEqual(tracker.health(1.08).source_age_s, 0.10)


class MachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config(
            preflight_stable_s=0.10,
            overlap_s=0.10,
            output_grace_s=0.01,
            migration_timeout_s=0.20,
            contract_ramp_s=0.20,
            candidate_hold_s=0.10,
            return_ramp_s=0.20,
            return_settle_s=0.10,
            official_verify_s=0.10,
            vibration_dwell_s=0.05,
        )
        self.machine = Machine(self.cfg)
        self.business = observation()
        self.develop = observation(system_state="Develop_MC")

    def reach_overlap(self) -> None:
        self.machine.step(0.0, self.business)
        directive = self.machine.step(0.11, self.business, Command.START)
        self.assertEqual(directive.state, State.OVERLAP_OFFICIAL)

    def reach_wait_develop(self) -> None:
        self.reach_overlap()
        directive = self.machine.step(0.22, self.business)
        self.assertEqual(directive.transition_request, "Develop_MC")
        self.assertEqual(directive.state, State.WAIT_DEVELOP)

    def reach_candidate_hold(self) -> float:
        self.reach_wait_develop()
        self.machine.step(0.23, self.develop)
        self.machine.step(0.34, self.develop)
        directive = self.machine.step(0.44, self.develop)
        self.assertEqual(directive.state, State.CANDIDATE_HOLD)
        return 0.44

    def reach_active(self) -> float:
        now = self.reach_candidate_hold()
        directive = self.machine.step(now + 0.11, self.develop, Command.ACTIVATE)
        self.assertEqual(directive.state, State.ACTIVE)
        return now + 0.11

    def test_preflight_rejects_stale_source_even_if_locally_fresh(self) -> None:
        stale = replace(healthy(), source_age_s=0.20)
        obs = observation(feedback=stale)
        self.machine.step(0.0, obs)
        directive = self.machine.step(0.2, obs, Command.START)
        self.assertEqual(directive.state, State.OBSERVE_OFFICIAL)
        self.assertIn("source_sample_stale", directive.reason)

    def test_preflight_rejects_target_mismatch(self) -> None:
        obs = observation(target_error_max_rad=0.06)
        self.machine.step(0.0, obs)
        directive = self.machine.step(0.2, obs, Command.START)
        self.assertIn("target_mismatch", directive.reason)

    def test_overlap_failure_before_request_is_safe_abort(self) -> None:
        self.reach_overlap()
        bad = observation(official=StreamHealth.missing())
        directive = self.machine.step(0.13, bad)
        self.assertEqual(directive.state, State.ABORTED)
        self.assertFalse(directive.publish)
        self.assertTrue(directive.can_exit)

    def test_migration_intent_requires_continuous_overlap(self) -> None:
        self.reach_overlap()
        bad_output = OutputContinuity(True, True, 0.001, 0.020, True)
        directive = self.machine.step(0.22, observation(output=bad_output))
        self.assertEqual(directive.state, State.ABORTED)
        self.assertIsNone(directive.transition_request)

    def test_rejected_migration_latches_and_keeps_publishing(self) -> None:
        self.reach_wait_develop()
        directive = self.machine.step(0.23, self.business, Command.MIGRATION_REJECTED)
        self.assertEqual(directive.state, State.LOCKED_RECOVERY)
        self.assertTrue(directive.publish)
        self.assertFalse(directive.can_exit)
        self.assertEqual(directive.target_mode, TargetMode.FREEZE_LAST_ACCEPTED)

    def test_develop_timeout_latches_instead_of_releasing(self) -> None:
        self.reach_wait_develop()
        directive = self.machine.step(0.43, self.business)
        self.assertEqual(directive.state, State.LOCKED_RECOVERY)
        self.assertTrue(directive.publish)

    def test_contract_ramp_is_monotonic_and_c2(self) -> None:
        self.reach_wait_develop()
        self.machine.step(0.23, self.develop)
        values = [self.machine.step(t, self.develop).contract_blend for t in (0.24, 0.28, 0.33, 0.38)]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(smoothstep5(0.0), 0.0)
        self.assertAlmostEqual(smoothstep5(1.0), 1.0)
        self.assertAlmostEqual(smoothstep5(0.5), 0.5)

    def test_activation_requires_fresh_stable_sonic(self) -> None:
        now = self.reach_candidate_hold()
        stale = replace(healthy(), source_age_s=0.5)
        directive = self.machine.step(
            now + 0.11,
            replace(self.develop, sonic=stale),
            Command.ACTIVATE,
        )
        self.assertEqual(directive.state, State.CANDIDATE_HOLD)
        self.assertIn("activate_rejected", directive.reason)

    def test_sonic_timeout_during_active_latches_without_auto_resume(self) -> None:
        now = self.reach_active()
        stale = replace(healthy(), local_age_s=0.5)
        directive = self.machine.step(now + 0.01, replace(self.develop, sonic=stale))
        self.assertEqual(directive.state, State.LOCKED_RECOVERY)
        recovered = self.machine.step(now + 0.02, self.develop)
        self.assertEqual(recovered.state, State.LOCKED_RECOVERY)

    def test_feedback_loss_keeps_last_command_instead_of_silent_exit(self) -> None:
        now = self.reach_active()
        directive = self.machine.step(
            now + 0.01,
            replace(self.develop, feedback=StreamHealth.missing()),
        )
        self.assertEqual(directive.state, State.LOCKED_RECOVERY)
        self.assertTrue(directive.publish)
        self.assertEqual(directive.target_mode, TargetMode.FREEZE_LAST_ACCEPTED)

    def test_vibration_faults_even_with_small_position_error(self) -> None:
        now = self.reach_active()
        vibration = replace(
            self.develop,
            tracking_error_max_rad=0.01,
            vibration_accel_rms_rad_s2=20.0,
        )
        first = self.machine.step(now + 0.01, vibration)
        self.assertEqual(first.state, State.ACTIVE)
        second = self.machine.step(now + 0.07, vibration)
        self.assertEqual(second.state, State.LOCKED_RECOVERY)
        self.assertEqual(second.reason, "vibration_detected")

    def test_exit_while_owned_cannot_drop_output(self) -> None:
        now = self.reach_active()
        directive = self.machine.step(now + 0.01, self.develop, Command.EXIT)
        self.assertEqual(directive.state, State.LOCKED_RECOVERY)
        self.assertTrue(directive.publish)
        self.assertFalse(directive.can_exit)

    def test_normal_handback_ramps_to_zero_before_ready_intent(self) -> None:
        now = self.reach_active()
        start = self.machine.step(now + 0.01, self.develop, Command.HANDBACK)
        self.assertEqual(start.state, State.RETURN_RAMP)
        mid = self.machine.step(now + 0.11, self.develop)
        self.assertGreater(mid.contract_blend, 0.0)
        self.assertLess(mid.contract_blend, 1.0)
        self.machine.step(now + 0.22, self.develop)
        ready = self.machine.step(now + 0.33, self.develop)
        self.assertEqual(ready.transition_request, "Ready")
        self.assertEqual(ready.contract_blend, 0.0)

    def test_handback_requires_post_marker_official_samples_and_dwell(self) -> None:
        now = self.reach_active()
        self.machine.step(now + 0.01, self.develop, Command.HANDBACK)
        self.machine.step(now + 0.22, self.develop)
        self.machine.step(now + 0.33, self.develop)
        ready_state = observation(system_state="Ready")
        directive = self.machine.step(now + 0.34, ready_state)
        self.assertEqual(directive.state, State.VERIFY_OFFICIAL)
        self.assertEqual(directive.target_mode, TargetMode.FREEZE_LAST_ACCEPTED)
        old_official = observation(official_sample_after_handback=False)
        directive = self.machine.step(now + 0.50, old_official)
        self.assertEqual(directive.state, State.VERIFY_OFFICIAL)
        self.assertEqual(directive.target_mode, TargetMode.FREEZE_LAST_ACCEPTED)
        fresh_official = observation(official_sample_after_handback=True)
        directive = self.machine.step(now + 0.51, fresh_official)
        self.assertEqual(directive.target_mode, TargetMode.MIRROR_LIVE_OFFICIAL)
        directive = self.machine.step(now + 0.62, fresh_official)
        self.assertEqual(directive.state, State.COMPLETE)
        self.assertTrue(directive.can_exit)
        self.assertFalse(directive.publish)

    def test_estop_is_only_immediate_release_after_ownership(self) -> None:
        self.reach_wait_develop()
        directive = self.machine.step(0.23, self.business, Command.ESTOP)
        self.assertEqual(directive.state, State.ESTOP)
        self.assertTrue(directive.can_exit)
        self.assertFalse(directive.publish)

    def test_module_has_no_robot_or_network_imports(self) -> None:
        source = Path(inspect.getsourcefile(Machine)).read_text(encoding="utf-8")
        for forbidden in ("import rclpy", "import socket", "import subprocess", "aimdk_msgs"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
