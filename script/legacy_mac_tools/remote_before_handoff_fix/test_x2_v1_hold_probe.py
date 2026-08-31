from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from x2_hal_guard import JOINT_NAMES, LOWER_LIMITS, UPPER_LIMITS
from x2_v1_hold_probe import (
    GROUPS,
    HandoffLifecycle,
    VibrationMonitor,
    anchor_problem,
    blended_torque_equivalent_profile,
    hold_drift_problem,
    is_official_command_sequence,
    official_handoff_problem,
    official_profile_change_problem,
    official_profile_fresh_problem,
    official_profile_problem,
    official_target_problem,
    relay_status_problem,
    suspended_target_problem,
    takeover_target_match_required,
    torque_equivalent_profile,
)
from x2_pd_contract import X2PdContract


class AnchorProblemTests(unittest.TestCase):
    def test_vibration_monitor_rejects_stationary_noise(self) -> None:
        monitor = VibrationMonitor()
        problem = None
        for index in range(351):
            t = index / 1000.0
            q = np.asarray([0.001 * np.sin(2.0 * np.pi * 12.0 * t)])
            dq = np.asarray([0.075 * np.cos(2.0 * np.pi * 12.0 * t)])
            problem = monitor.update("test", ["joint"], q, dq, t)
        self.assertIsNone(problem)

    def test_vibration_monitor_detects_sustained_oscillation(self) -> None:
        monitor = VibrationMonitor()
        problem = None
        for index in range(351):
            t = index / 1000.0
            q = np.asarray([0.012 * np.sin(2.0 * np.pi * 10.0 * t)])
            dq = np.asarray([
                0.012 * 2.0 * np.pi * 10.0 * np.cos(2.0 * np.pi * 10.0 * t)
            ])
            problem = monitor.update("test", ["joint"], q, dq, t)
        self.assertIsNotNone(problem)
        self.assertIn("vibration joint", problem)
        self.assertIn("reversals=", problem)

    def test_vibration_monitor_rejects_one_way_motion(self) -> None:
        monitor = VibrationMonitor()
        problem = None
        for index in range(351):
            t = index / 1000.0
            q = np.asarray([0.3 * t])
            dq = np.asarray([0.3])
            problem = monitor.update("test", ["joint"], q, dq, t)
        self.assertIsNone(problem)

    def test_rt_relay_replaces_python_500hz_timer(self) -> None:
        source = Path(__file__).with_name("x2_v1_hold_probe.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("node.create_timer(0.002, control_tick)", source)
        self.assertIn('name="x2_candidate_command_loop"', source)
        relay_source = (
            Path(__file__).with_name("x2_rt_relay")
            / "src"
            / "x2_rt_command_relay.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("constexpr auto kPublishPeriod = 2ms", relay_source)
        self.assertIn("publish_thread_", relay_source)

    def test_rt_relay_status_accepts_stable_500hz_publish(self) -> None:
        status = {
            "ready": True,
            "active": True,
            "fault": False,
            "period_p95_ms": 2.2,
            "period_max_ms": 4.0,
        }
        self.assertIsNone(relay_status_problem(status))

    def test_rt_relay_status_rejects_fault_and_jitter(self) -> None:
        faulted = {
            "ready": True,
            "active": True,
            "fault": True,
            "period_p95_ms": 2.0,
            "period_max_ms": 2.5,
        }
        self.assertIn("fault", relay_status_problem(faulted))
        jitter = dict(faulted, fault=False, period_max_ms=8.1)
        self.assertIn("maximum period", relay_status_problem(jitter))

    def test_state_services_use_a_dedicated_ros_node(self) -> None:
        source = Path(__file__).with_name("x2_v1_hold_probe.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('service_node = Node("x2_v1_hold_probe_services")', source)
        self.assertIn("rclpy.spin_once(service_node", source)
        self.assertIn("get_client = service_node.create_client", source)
        self.assertIn("migrate_client = service_node.create_client", source)
        self.assertIn("for attempt in range(1, 4)", source)

    def test_single_joint_sonic_gain_conversion_preserves_torque(self) -> None:
        profile = {name: (0.0, 0.0, 0.0, 20.0, 1.0) for name in JOINT_NAMES}
        name = "left_shoulder_pitch_joint"
        index = JOINT_NAMES.index(name)
        profile[name] = (0.4, 0.03, 0.7, 30.0, 2.0)
        q = np.zeros(31)
        dq = np.zeros(31)
        q[index] = 0.35
        dq[index] = 0.01
        contract = X2PdContract.build(
            names=JOINT_NAMES,
            default_position=np.zeros(31),
            action_scale=np.ones(31),
            stiffness=np.full(31, 15.0),
            damping=np.full(31, 0.8),
            source="unit-test",
        )
        candidate, report = torque_equivalent_profile(
            profile, q, dq, contract, (name,)
        )
        self.assertEqual(candidate[JOINT_NAMES[0]], profile[JOINT_NAMES[0]])
        target, velocity, effort, stiffness, damping = candidate[name]
        official_tau = 0.7 + 30.0 * (0.4 - 0.35) + 2.0 * (0.03 - 0.01)
        candidate_tau = effort + stiffness * (target - q[index]) + damping * (
            velocity - dq[index]
        )
        self.assertAlmostEqual(candidate_tau, official_tau, places=12)
        self.assertEqual((velocity, effort, stiffness, damping), (0.0, 0.0, 15.0, 0.8))
        self.assertLessEqual(report["max_torque_residual_nm"], 1e-12)

    def test_full_body_sonic_gain_conversion_preserves_every_joint(self) -> None:
        measured = (LOWER_LIMITS + UPPER_LIMITS) * 0.5
        profile = {
            name: (float(measured[index] + 0.001), 0.0, 0.1, 30.0, 2.0)
            for index, name in enumerate(JOINT_NAMES)
        }
        contract = X2PdContract.build(
            names=JOINT_NAMES,
            default_position=measured,
            action_scale=np.ones(31),
            stiffness=np.full(31, 20.0),
            damping=np.ones(31),
            source="unit-test",
        )
        candidate, report = torque_equivalent_profile(
            profile,
            measured,
            np.zeros(31),
            contract,
            tuple(JOINT_NAMES),
        )
        self.assertEqual(len(report["converted"]), 31)
        self.assertLessEqual(report["max_torque_residual_nm"], 1e-12)
        for index, name in enumerate(JOINT_NAMES):
            target, velocity, effort, stiffness, damping = candidate[name]
            expected_torque = 0.1 + 30.0 * 0.001
            actual_torque = (
                effort
                + stiffness * (target - measured[index])
                + damping * (velocity - 0.0)
            )
            self.assertAlmostEqual(actual_torque, expected_torque, places=12)

    def test_gain_ramp_preserves_torque_at_every_stage(self) -> None:
        measured = (LOWER_LIMITS + UPPER_LIMITS) * 0.5
        velocity = np.linspace(-0.02, 0.02, 31)
        profile = {
            name: (
                float(measured[index] + 0.001),
                0.0,
                0.1,
                30.0,
                2.0,
            )
            for index, name in enumerate(JOINT_NAMES)
        }
        contract = X2PdContract.build(
            names=JOINT_NAMES,
            default_position=measured,
            action_scale=np.ones(31),
            stiffness=np.full(31, 20.0),
            damping=np.ones(31),
            source="unit-test",
        )
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            candidate, report = blended_torque_equivalent_profile(
                profile, measured, velocity, contract, alpha
            )
            self.assertAlmostEqual(report["alpha"], alpha)
            self.assertLessEqual(report["max_torque_residual_nm"], 1e-12)
            for index, name in enumerate(JOINT_NAMES):
                target, command_velocity, effort, kp, kd = candidate[name]
                expected = 0.1 + 30.0 * 0.001 - 2.0 * velocity[index]
                actual = (
                    effort
                    + kp * (target - measured[index])
                    + kd * (command_velocity - velocity[index])
                )
                self.assertAlmostEqual(actual, expected, places=12)
                if alpha == 1.0:
                    self.assertEqual(kp, contract.stiffness[index])
                    self.assertEqual(kd, contract.damping[index])

    def test_gain_ramp_rejects_invalid_alpha(self) -> None:
        measured = (LOWER_LIMITS + UPPER_LIMITS) * 0.5
        profile = {
            name: (float(measured[index]), 0.0, 0.0, 30.0, 2.0)
            for index, name in enumerate(JOINT_NAMES)
        }
        contract = X2PdContract.build(
            names=JOINT_NAMES,
            default_position=measured,
            action_scale=np.ones(31),
            stiffness=np.full(31, 20.0),
            damping=np.ones(31),
            source="unit-test",
        )
        with self.assertRaisesRegex(ValueError, "alpha"):
            blended_torque_equivalent_profile(
                profile, measured, np.zeros(31), contract, 1.01
            )

    def test_live_torque_step_keeps_official_profile_through_migration(self) -> None:
        source = Path(__file__).with_name("x2_v1_hold_probe.py").read_text(
            encoding="utf-8"
        )
        mirror = source.index("apply_mirror_profile(\n            official_capture")
        migration = source.index('migrate("Develop_MC")')
        ramp = source.index('"[v1-hold] GAIN RAMP ACTIVE:')
        step = source.index('f"[v1-hold] STEP ACTIVE:')
        self.assertLess(mirror, migration)
        self.assertLess(migration, ramp)
        self.assertLess(ramp, step)

    def test_live_gain_ramp_checks_feedback_relay_and_motion_faults(self) -> None:
        source = Path(__file__).with_name("x2_v1_hold_probe.py").read_text(
            encoding="utf-8"
        )
        start = source.index("ramp_start = time.monotonic()")
        end = source.index("for group, (base, _, _) in GROUPS.items()", start)
        ramp = source[start:end]
        self.assertIn('if motion["fault"] is not None:', ramp)
        self.assertIn("current = snapshot(0.08)", ramp)
        self.assertIn("blended_torque_equivalent_profile(", ramp)
        self.assertIn("apply_gain_ramp_profile(ramp_profile)", ramp)
        self.assertIn("wait_relay_ready(1.0, require_active=True)", ramp)
        self.assertIn("alpha >= 1.0", ramp)

    def test_torque_conversion_rejects_out_of_limit_candidate(self) -> None:
        profile = {name: (0.0, 0.0, 0.0, 20.0, 1.0) for name in JOINT_NAMES}
        name = "left_shoulder_pitch_joint"
        profile[name] = (2.0, 0.0, 0.0, 100.0, 1.0)
        contract = X2PdContract.build(
            names=JOINT_NAMES,
            default_position=np.zeros(31),
            action_scale=np.ones(31),
            stiffness=np.ones(31),
            damping=np.ones(31),
            source="unit-test",
        )
        with self.assertRaisesRegex(ValueError, "outside model limit"):
            torque_equivalent_profile(
                profile, np.zeros(31), np.zeros(31), contract, (name,)
            )

    def test_target_match_policy_keeps_normal_takeover_strict(self) -> None:
        self.assertTrue(takeover_target_match_required(False, False))
        self.assertFalse(takeover_target_match_required(True, False))
        self.assertFalse(takeover_target_match_required(False, True))

    def test_mirror_only_can_release_without_migration(self) -> None:
        lifecycle = HandoffLifecycle()
        self.assertTrue(lifecycle.may_release_mirror())
        self.assertFalse(lifecycle.recovery_required())

    def test_direct_migration_rejection_requires_recovery(self) -> None:
        lifecycle = HandoffLifecycle(migration_requested=True)
        self.assertFalse(lifecycle.may_release_mirror())
        self.assertTrue(lifecycle.recovery_required())

    def test_service_timeout_keeps_mirror_owned(self) -> None:
        lifecycle = HandoffLifecycle(migration_requested=True, develop_confirmed=False)
        self.assertFalse(lifecycle.may_release_mirror())

    def test_develop_feedback_outage_keeps_mirror_owned(self) -> None:
        lifecycle = HandoffLifecycle(migration_requested=True, develop_confirmed=True)
        self.assertTrue(lifecycle.recovery_required())
        self.assertFalse(lifecycle.may_release_mirror())

    def test_fault_during_motion_keeps_mirror_until_standing(self) -> None:
        lifecycle = HandoffLifecycle(migration_requested=True, develop_confirmed=True)
        self.assertFalse(lifecycle.may_release_mirror())
        lifecycle.standing_verified = True
        self.assertTrue(lifecycle.may_release_mirror())

    def test_estop_is_the_only_non_standing_release_authority(self) -> None:
        lifecycle = HandoffLifecycle(migration_requested=True, develop_confirmed=True)
        lifecycle.estop_confirmed = True
        self.assertTrue(lifecycle.may_release_mirror())

    def test_all_joint_groups_are_complete(self) -> None:
        self.assertEqual(set(GROUPS), {"leg", "waist", "arm", "head"})

    def test_accepts_unchanged_valid_pose(self) -> None:
        pose = np.zeros(31)
        pose[3] = 0.5
        pose[9] = 0.5
        self.assertIsNone(anchor_problem(pose, pose.copy()))

    def test_rejects_migration_pose_jump(self) -> None:
        before = np.zeros(31)
        after = before.copy()
        after[15] = 0.051
        self.assertIn("pose changed", anchor_problem(after, before))

    def test_rejects_bad_shape(self) -> None:
        self.assertIn("invalid", anchor_problem(np.zeros(29)))

    def test_accepts_complete_powered_official_profile(self) -> None:
        profile = {name: (0.0, 0.0, 0.0, 20.0, 1.0) for name in JOINT_NAMES}
        self.assertIsNone(official_profile_problem(profile))

    def test_rejects_zero_stiffness_official_profile(self) -> None:
        profile = {name: (0.0, 0.0, 0.0, 20.0, 1.0) for name in JOINT_NAMES}
        profile[JOINT_NAMES[15]] = (0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertIn("enter official Standing", official_profile_problem(profile))

    def test_rejects_incomplete_official_profile(self) -> None:
        profile = {
            name: (0.0, 0.0, 0.0, 20.0, 1.0) for name in JOINT_NAMES[:-1]
        }
        self.assertIn("profile mismatch", official_profile_problem(profile))

    def test_rejects_moving_official_profile(self) -> None:
        profile = {name: (0.0, 0.0, 0.0, 20.0, 1.0) for name in JOINT_NAMES}
        profile[JOINT_NAMES[15]] = (0.0, 0.11, 0.0, 20.0, 1.0)
        self.assertIn("not stationary", official_profile_problem(profile))

    def test_official_target_must_match_measured_pose(self) -> None:
        profile = {name: (0.0, 0.0, 0.0, 20.0, 1.0) for name in JOINT_NAMES}
        measured = np.zeros(31)
        self.assertIsNone(official_target_problem(profile, measured))
        profile[JOINT_NAMES[3]] = (0.151, 0.0, 4.0, 20.0, 1.0)
        problem = official_target_problem(profile, measured)
        self.assertIn("differs", problem)
        self.assertIn("target=0.151rad measured=0.000rad", problem)

    def test_suspended_target_offset_has_absolute_limit(self) -> None:
        profile = {name: (0.0, 0.0, 0.0, 20.0, 1.0) for name in JOINT_NAMES}
        measured = np.zeros(31)
        profile[JOINT_NAMES[10]] = (0.299, 0.0, 0.0, 20.0, 1.0)
        self.assertIsNone(suspended_target_problem(profile, measured))
        profile[JOINT_NAMES[10]] = (0.301, 0.0, 0.0, 20.0, 1.0)
        self.assertIn("too large", suspended_target_problem(profile, measured))

    def test_official_profile_stability_rejects_target_change(self) -> None:
        before = {name: (0.0, 0.0, 0.0, 20.0, 1.0) for name in JOINT_NAMES}
        after = dict(before)
        self.assertIsNone(official_profile_change_problem(before, after))
        after[JOINT_NAMES[15]] = (0.011, 0.0, 0.0, 20.0, 1.0)
        self.assertIn("target is changing", official_profile_change_problem(before, after))

    def test_official_profile_stability_rejects_gain_change(self) -> None:
        before = {name: (0.0, 0.0, 0.0, 20.0, 1.0) for name in JOINT_NAMES}
        after = dict(before)
        after[JOINT_NAMES[0]] = (0.0, 0.0, 0.0, 22.0, 1.0)
        self.assertIn("stiffness is changing", official_profile_change_problem(before, after))

    def test_official_sequence_marker(self) -> None:
        self.assertTrue(is_official_command_sequence(1))
        self.assertTrue(is_official_command_sequence(462977))
        self.assertFalse(is_official_command_sequence(0))

    def test_fresh_official_profile_requires_post_handoff_samples(self) -> None:
        profile = {name: (0.0, 0.0, 0.0, 20.0, 1.0) for name in JOINT_NAMES}
        times = {group: 10.0 for group in GROUPS}
        self.assertIsNone(official_profile_fresh_problem(profile, times, 9.0, 10.05))
        times["arm"] = 8.0
        self.assertIn(
            "predates handback",
            official_profile_fresh_problem(profile, times, 9.0, 10.05),
        )

    def test_fresh_official_profile_rejects_stale_group(self) -> None:
        profile = {name: (0.0, 0.0, 0.0, 20.0, 1.0) for name in JOINT_NAMES}
        times = {group: 10.0 for group in GROUPS}
        times["leg"] = 9.8
        self.assertIn(
            "stale",
            official_profile_fresh_problem(profile, times, 9.0, 10.05),
        )

    def test_handoff_requires_business_and_fresh_powered_commands(self) -> None:
        profile = {name: (0.0, 0.0, 0.0, 20.0, 1.0) for name in JOINT_NAMES}
        times = {group: 10.0 for group in GROUPS}
        self.assertIsNone(
            official_handoff_problem("Business", 1, 1, profile, times, 9.0, 10.05)
        )
        self.assertIn(
            "not stable Business",
            official_handoff_problem("Ready", 1, 1, profile, times, 9.0, 10.05),
        )
        profile[JOINT_NAMES[0]] = (0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertIn(
            "enter official Standing",
            official_handoff_problem("Business", 1, 1, profile, times, 9.0, 10.05),
        )

    def test_hold_drift_guard(self) -> None:
        target = np.zeros(31)
        actual = target.copy()
        actual[10] = 0.031
        self.assertIn("hold drift", hold_drift_problem(actual, target))
        actual[10] = 0.029
        self.assertIsNone(hold_drift_problem(actual, target))

    def test_arm_motion_does_not_trip_anchored_body_guard(self) -> None:
        target = np.zeros(31)
        actual = target.copy()
        actual[15] = 0.05
        self.assertIsNone(hold_drift_problem(actual, target, arms_active=True))
        self.assertIn("hold drift", hold_drift_problem(actual, target))


if __name__ == "__main__":
    unittest.main()
