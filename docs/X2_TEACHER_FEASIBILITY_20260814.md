# X2 direct-teacher feasibility gate (2026-08-14)

## Scope

This gate tested whether the two missing command roles in the X2 scratch
pipeline could be supplied without another blind student-training run.  All
rollouts used the identified Session03/Session04 actuator-response domain,
the sole12 collision profile, disabled self-collisions, and the direct
normalized 15-joint target contract.

No scratch checkpoint was trained or loaded.  The command-feedback screen
used the frozen Stage219 policy only as a proposal teacher.  The direct CEM
screen loaded no policy weights and used the official zero-mean gait cycle
only as a weak periodic prior.

## Results

### Bounded command feedback (v8)

Decision: `FAIL_COMMAND_FEEDBACK_ESCALATE_TO_DIRECT_15D_TEACHER`.

- Low-speed candidates could often survive and track 0.20 m/s, but every
  candidate inherited about 15% proposal-action saturation.
- The best stable right-turn candidate reached only about -0.10 rad/s for a
  -0.30 rad/s target.
- Teacher-data collection and student training remained locked.

### Direct 15-D gait-feedback CEM (v9)

Decision: `FAIL_PARAMETRIC_TEACHER_REQUIRE_CLONE_STATE_MPC_OR_RL`.

Independent 400-step verification used 256 lanes per role:

| Role | Survived | Mean vx | Mean yaw rate | Required command |
|---|---:|---:|---:|---:|
| `vx_0p20` | 200/256 | 0.0255 m/s | 0.0014 rad/s | 0.20 m/s, 0.00 rad/s |
| `turn_right` | 194/256 | 0.0474 m/s | -0.0021 rad/s | 0.35 m/s, -0.30 rad/s |

The direct actions had zero saturation in verification.  The failure is
therefore not explained by the normalized action bound.  A zero-mean
kinematic cycle plus compact linear body feedback did not create the contact
switching and momentum regulation needed for a safe locomotion teacher.

## Decision and next gate

- Do not enlarge either gain grid.
- Do not collect demonstrations from these candidates.
- Do not launch another scratch-student update from these candidates.
- First prove that articulation state, joint state, actuator delay buffers,
  actuator low-pass state, and action history can be cloned across Isaac
  environments with bounded rollout divergence.
- Only after that audit passes, evaluate a short-horizon direct-action MPC
  teacher.  If exact response-state cloning is not feasible, use an online RL
  teacher in the identified response domain instead of claiming MPC evidence.

Raw logs and JSON results live under the task-card-compliant external data
root.  Their sizes, SHA256 values, source commit, and recovery mapping are in
`manifests/x2_teacher_feasibility_20260814.json`.
