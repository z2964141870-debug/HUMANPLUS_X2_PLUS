from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "X2_sonic_real"))

from x2_command_consumption import CommandConsumptionMonitor  # noqa: E402
from x2_hal_guard import JOINT_NAMES  # noqa: E402
from x2_independent_supervisor import (  # noqa: E402
    IndependentSupervisor,
    SupervisorSnapshot,
)
from x2_pd_contract import GROUP_LAYOUT, X2PdContract  # noqa: E402

from mc_transition_next.offline_adapter import (  # noqa: E402
    AdapterInput,
    OfflineTransitionAdapter,
    PdFrame,
)
from mc_transition_next.state_machine import (  # noqa: E402
    Command,
    Config,
    Machine,
    Observation,
    OutputContinuity,
    State,
    StreamHealth,
    TargetMode,
)


def healthy(*, source_age_s: float = 0.003, local_age_s: float = 0.002) -> StreamHealth:
    return StreamHealth(
        complete=True,
        sequence_ok=True,
        local_age_s=local_age_s,
        source_age_s=source_age_s,
        group_skew_s=0.001,
        max_gap_s=0.004,
        stable_for_s=5.0,
    )


def continuity(*, max_gap_s: float = 0.003) -> OutputContinuity:
    return OutputContinuity(True, True, 0.001, max_gap_s, True)


def observation(system_state: str = "Business", **changes) -> Observation:
    base = Observation(
        system_state=system_state,
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


class OfflineAdapterReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = X2PdContract.build(
            names=JOINT_NAMES,
            default_position=np.zeros(31),
            action_scale=np.full(31, 0.2),
            stiffness=np.full(31, 20.0),
            damping=np.full(31, 1.0),
            source="offline-adapter-test",
        )
        self.monitor = CommandConsumptionMonitor()
        cfg = Config(
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
        self.machine = Machine(cfg)
        self.adapter = OfflineTransitionAdapter(
            machine=self.machine,
            contract=self.contract,
            supervisor=IndependentSupervisor(self.contract.fingerprint),
            consumption_monitor=self.monitor,
        )
        self.anchor = np.linspace(-0.10, 0.10, 31)
        self.official = PdFrame.build(
            target=self.anchor,
            velocity=np.zeros(31),
            effort=np.zeros(31),
            stiffness=np.full(31, 10.0),
            damping=np.full(31, 0.5),
            contract_fingerprint=self.contract.fingerprint,
        )
        self.action = np.linspace(-0.25, 0.25, 31)

    def prove_consumption(self, now_s: float) -> None:
        for group, (_, length) in GROUP_LAYOUT.items():
            self.monitor.set_matched_subscriptions(group, 1)
            baseline = np.zeros(length)
            state = baseline.copy()
            state[0] = 0.004
            self.monitor.record_command(group, now_s - 0.002)
            self.monitor.start_probe(
                group,
                local_index=0,
                delta_rad=0.02,
                baseline=baseline,
                now_s=now_s - 0.003,
            )
            self.monitor.record_state(group, state, now_s - 0.001)

    def data(
        self,
        now_s: float,
        obs: Observation | None = None,
        *,
        official: PdFrame | None = None,
        controller_seen_s: float | None = None,
        measured_position: np.ndarray | None = None,
        measured_velocity: np.ndarray | None = None,
    ) -> AdapterInput:
        seen = now_s - 0.001
        return AdapterInput(
            observation=obs or observation(),
            supervisor_snapshot=SupervisorSnapshot(
                now_s=now_s,
                state_seen_s=seen,
                command_seen_s=seen,
                policy_seen_s=seen,
                controller_seen_s=(
                    seen if controller_seen_s is None else controller_seen_s
                ),
                contract_fingerprint=self.contract.fingerprint,
                consumers_proven=False,
                tracking_error_rad=0.01,
                tilt_rad=0.02,
                angular_velocity_rad_s=0.03,
                joint_velocity_rms_rad_s=0.04,
                ownership_active=False,
                suspended=True,
            ),
            official_frame=official or self.official,
            measured_position=(
                np.zeros(31) if measured_position is None else measured_position
            ),
            measured_velocity=(
                np.zeros(31) if measured_velocity is None else measured_velocity
            ),
            sonic_action=self.action,
        )

    def tick(
        self,
        now_s: float,
        obs: Observation | None = None,
        command: Command = Command.NONE,
        **data_changes,
    ):
        self.prove_consumption(now_s)
        return self.adapter.step(
            now_s, self.data(now_s, obs, **data_changes), command
        )

    def reach_wait_develop(self) -> float:
        self.tick(1.00)
        started = self.tick(1.11, command=Command.START)
        self.assertEqual(started.directive.state, State.OVERLAP_OFFICIAL)
        requested = self.tick(1.22)
        self.assertEqual(requested.directive.transition_request, "Develop_MC")
        return 1.22

    def reach_candidate_hold(self) -> float:
        self.reach_wait_develop()
        self.tick(1.23, observation("Develop_MC"))
        held = self.tick(1.44, observation("Develop_MC"))
        self.assertEqual(held.directive.state, State.CANDIDATE_HOLD)
        return 1.44

    def reach_active(self) -> float:
        self.reach_candidate_hold()
        active = self.tick(1.55, observation("Develop_MC"), Command.ACTIVATE)
        self.assertEqual(active.directive.state, State.ACTIVE)
        return 1.55

    def test_31_joint_pd_fields_blend_atomically(self) -> None:
        self.reach_wait_develop()
        self.tick(1.23, observation("Develop_MC"))
        midpoint = self.tick(1.33, observation("Develop_MC"))
        self.assertEqual(midpoint.directive.target_mode, TargetMode.CANDIDATE_ANCHOR)
        self.assertAlmostEqual(midpoint.directive.contract_blend, 0.5)
        np.testing.assert_allclose(midpoint.frame.target, self.anchor * (10.0 / 15.0))
        np.testing.assert_allclose(midpoint.frame.velocity, np.zeros(31))
        np.testing.assert_allclose(midpoint.frame.effort, np.zeros(31))
        np.testing.assert_allclose(midpoint.frame.stiffness, np.full(31, 15.0))
        np.testing.assert_allclose(midpoint.frame.damping, np.full(31, 0.75))
        official_torque = self.official.stiffness * self.official.target
        midpoint_torque = midpoint.frame.stiffness * midpoint.frame.target
        np.testing.assert_allclose(midpoint_torque, official_torque)
        self.assertEqual(midpoint.frame.target.shape, (31,))
        with self.assertRaises(ValueError):
            midpoint.frame.target.setflags(write=True)

    def test_nonzero_velocity_and_effort_preserve_official_torque(self) -> None:
        official = PdFrame.build(
            target=self.anchor,
            velocity=np.linspace(-0.03, 0.03, 31),
            effort=np.linspace(-0.5, 0.5, 31),
            stiffness=np.full(31, 10.0),
            damping=np.full(31, 0.5),
            contract_fingerprint=self.contract.fingerprint,
        )
        q = np.linspace(-0.04, 0.04, 31)
        dq = np.linspace(0.02, -0.02, 31)
        inputs = {
            "official": official,
            "measured_position": q,
            "measured_velocity": dq,
        }
        self.tick(1.00, **inputs)
        self.tick(1.11, command=Command.START, **inputs)
        self.tick(1.22, **inputs)
        self.tick(1.23, observation("Develop_MC"), **inputs)
        midpoint = self.tick(1.33, observation("Develop_MC"), **inputs)
        official_torque = (
            official.effort
            + official.stiffness * (official.target - q)
            + official.damping * (official.velocity - dq)
        )
        candidate_torque = (
            midpoint.frame.effort
            + midpoint.frame.stiffness * (midpoint.frame.target - q)
            + midpoint.frame.damping * (midpoint.frame.velocity - dq)
        )
        np.testing.assert_allclose(candidate_torque, official_torque, atol=1e-12)
        np.testing.assert_allclose(midpoint.frame.velocity, official.velocity * 0.5)
        np.testing.assert_allclose(midpoint.frame.effort, official.effort * 0.5)

    def test_zero_action_holds_torque_equivalent_candidate_anchor(self) -> None:
        self.reach_candidate_hold()
        self.action = np.zeros(31)
        result = self.tick(
            1.55,
            observation("Develop_MC"),
            Command.ACTIVATE,
            official=self.official,
        )
        candidate_anchor = self.adapter._candidate_anchor.copy()
        np.testing.assert_allclose(result.frame.target, candidate_anchor)
        zero = replace(self.data(1.56, observation("Develop_MC")), sonic_action=np.zeros(31))
        self.prove_consumption(1.56)
        held = self.adapter.step(1.56, zero)
        np.testing.assert_allclose(held.frame.target, candidate_anchor)

    def test_sonic_is_offset_and_rate_limited_on_local_tick(self) -> None:
        now = self.reach_active()
        previous = self.adapter._last_accepted.target.copy()
        data = replace(
            self.data(now + 0.01, observation("Develop_MC")),
            sonic_action=np.full(31, 20.0),
        )
        self.prove_consumption(now + 0.01)
        result = self.adapter.step(now + 0.01, data)
        delta = np.abs(result.frame.target - previous)
        self.assertLessEqual(float(np.max(delta)), 0.002 + 1e-12)
        self.assertLessEqual(
            float(np.max(np.abs(result.frame.target - self.adapter._candidate_anchor))),
            0.03 + 1e-12,
        )

    def test_official_target_outside_model_limit_is_never_moved_farther_out(self) -> None:
        outside = self.official.target.copy()
        outside[16] = self.contract.lower_limit[16] - 0.05
        official = PdFrame.build(
            target=outside,
            velocity=self.official.velocity,
            effort=self.official.effort,
            stiffness=self.official.stiffness,
            damping=self.official.damping,
            contract_fingerprint=self.contract.fingerprint,
        )
        measured = outside.copy()
        inputs = {"official": official, "measured_position": measured}
        self.tick(1.00, **inputs)
        self.tick(1.11, command=Command.START, **inputs)
        self.tick(1.22, **inputs)
        self.tick(1.23, observation("Develop_MC"), **inputs)
        self.tick(1.44, observation("Develop_MC"), **inputs)
        self.tick(1.55, observation("Develop_MC"), Command.ACTIVATE, **inputs)
        outward = replace(
            self.data(
                1.56,
                observation("Develop_MC"),
                official=official,
                measured_position=measured,
            ),
            sonic_action=np.full(31, -20.0),
        )
        self.prove_consumption(1.56)
        result = self.adapter.step(1.56, outward)
        self.assertGreaterEqual(result.frame.target[16], outside[16])

    def test_source_stale_but_locally_fresh_blocks_start(self) -> None:
        self.tick(1.00)
        stale = replace(
            observation(), feedback=healthy(source_age_s=0.20, local_age_s=0.001)
        )
        result = self.tick(1.20, stale, Command.START)
        self.assertEqual(result.directive.state, State.OBSERVE_OFFICIAL)
        self.assertIn("source_sample_stale", result.directive.reason)

    def test_consumption_loss_after_develop_intent_latches_output(self) -> None:
        now = self.reach_wait_develop()
        self.monitor.set_matched_subscriptions("leg", 0)
        result = self.adapter.step(
            now + 0.01,
            self.data(now + 0.01, observation("Develop_MC")),
        )
        self.assertEqual(result.directive.state, State.LOCKED_RECOVERY)
        self.assertEqual(result.directive.target_mode, TargetMode.FREEZE_LAST_ACCEPTED)
        self.assertIn("command_consumption_not_proven", result.directive.reason)
        np.testing.assert_array_equal(result.frame.target, self.official.target)

    def test_vibration_latches_with_low_position_error(self) -> None:
        now = self.reach_active()
        vibration = observation(
            "Develop_MC",
            tracking_error_max_rad=0.01,
            vibration_accel_rms_rad_s2=20.0,
        )
        first = self.tick(now + 0.01, vibration)
        self.assertEqual(first.directive.state, State.ACTIVE)
        second = self.tick(now + 0.07, vibration)
        self.assertEqual(second.directive.state, State.LOCKED_RECOVERY)
        self.assertEqual(second.directive.reason, "vibration_detected")

    def test_output_gap_after_develop_intent_latches(self) -> None:
        now = self.reach_wait_develop()
        gap = observation(output=continuity(max_gap_s=0.020))
        result = self.tick(now + 0.02, gap)
        self.assertEqual(result.directive.state, State.LOCKED_RECOVERY)
        self.assertEqual(result.directive.reason, "output_publish_gap")
        self.assertTrue(result.directive.publish)

    def test_controller_process_gap_after_develop_intent_latches(self) -> None:
        now = self.reach_wait_develop()
        self.prove_consumption(now + 0.01)
        result = self.adapter.step(
            now + 0.01,
            self.data(
                now + 0.01,
                observation("Develop_MC"),
                controller_seen_s=now - 1.0,
            ),
        )
        self.assertEqual(result.directive.state, State.LOCKED_RECOVERY)
        self.assertIn("controller_stale", result.directive.reason)

    def test_handback_requires_verified_post_marker_official_frame(self) -> None:
        now = self.reach_active()
        self.tick(now + 0.01, observation("Develop_MC"), Command.HANDBACK)
        self.tick(now + 0.22, observation("Develop_MC"))
        ready_request = self.tick(now + 0.33, observation("Develop_MC"))
        self.assertEqual(ready_request.directive.transition_request, "Ready")
        entered_verify = self.tick(now + 0.34, observation("Ready"))
        self.assertEqual(entered_verify.directive.state, State.VERIFY_OFFICIAL)
        old = self.tick(now + 0.50, observation(official_sample_after_handback=False))
        self.assertEqual(old.directive.target_mode, TargetMode.FREEZE_LAST_ACCEPTED)

        live_official = PdFrame.build(
            target=np.full(31, 0.03),
            velocity=np.zeros(31),
            effort=np.zeros(31),
            stiffness=np.full(31, 12.0),
            damping=np.full(31, 0.6),
            contract_fingerprint=self.contract.fingerprint,
        )
        fresh = self.tick(
            now + 0.51,
            observation(official_sample_after_handback=True),
            official=live_official,
        )
        self.assertEqual(fresh.directive.target_mode, TargetMode.MIRROR_LIVE_OFFICIAL)
        np.testing.assert_array_equal(fresh.frame.target, live_official.target)
        complete = self.tick(
            now + 0.62,
            observation(official_sample_after_handback=True),
            official=live_official,
        )
        self.assertEqual(complete.directive.state, State.COMPLETE)
        self.assertIsNone(complete.frame)
        self.assertTrue(complete.directive.can_exit)


if __name__ == "__main__":
    unittest.main()
