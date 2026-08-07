# X2 Stage102 KDMR-lite / COM-aware Pose Repair — Final Verdict

## Decision

**REJECTED — do not train on the Stage102 diagnostic cache.**

Stage102 answered a narrow causal question: can independent per-frame,
bounded leg/waist pose residuals put the mass-weighted X2 COM inside the
currently labelled stance sole while preserving a usable time series?  The
answer is **no for the current Stage100 contact schedule**.  Static per-frame
feasibility is almost completely recoverable, but enforcing it at direct
left/right support switches creates discontinuities that fail the existing
kinematic gate by a large margin.

This is a useful negative result.  It rules out “project COM into the stance
foot on every frame” as the next training-data fix; it does not rule out
trajectory-level kinodynamic retargeting.

## Hypothesis

The Stage100-regularized reference has reasonable root/contact geometry, but
its single-support COM lies outside the labelled sole on roughly 90–100% of
frames.  A small residual over the 12 leg joints and 3 waist joints might move
the COM into the stance polygon without destroying the original motion.

## Intervention

- input: `motion_lib_x2/stage100_official_true_forward4_contact_geometry_regularized_v2`;
- MuJoCo mass parameters from `assets/agibot_x2/x2_ultra.xml`, total mass
  `43.474797 kg`;
- support rectangle derived from the ankle collision spheres:
  `x=[-0.070, 0.144] m`, `y=[-0.065, 0.065] m`;
- hard inward margin: `0.010 m`;
- optimized joints: both legs plus waist yaw/pitch/roll only;
- exact mirror-paired optimization: one canonical residual generates the
  mirrored residual by the named X2 joint permutation/sign map;
- objective retains stance-foot pose, swing-foot relative pose, upper-body
  keypoints, joint proximity and one-step residual smoothness;
- all joint limits and per-joint residual caps are hard bounds;
- no policy training and no source-cache overwrite.

## Control

The unmodified Stage100 regularized-v2 cache already passed its intended
contact/root gate:

- stance slip p95: at most `0.5445 m/s`;
- double-support inter-foot relative speed p95: at most `0.0341 m/s`;
- root acceleration p95: at most `14.9559 m/s²`;
- source maximum DOF step: `0.050000 rad`;
- source maximum root step: about `0.0189 m`.

## Result

### What worked locally

| base motion | SS frames | COM outside before | COM outside after | COM violation p95 before | COM violation max after | residual p95/max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B | 188 | 97.34% | 0% | 0.1050 m | 0.000000039 m | 0.1200 / 0.2114 rad |
| D | 236 | 93.64% | 0% | 0.1111 m | 0.000000025 m | 0.1343 / 0.2584 rad |

The strict mirror errors were exactly zero for DOF and root translation.
Upper-body keypoint relative-position p95 stayed below `9.4 mm`; stance-anchor
error p95 stayed below `6.5 mm`; swing-relative-position error p95 was
`52–62 mm`.

### What failed globally

The standard independent kinematic gate failed **all 4/4 motions**:

| check | gate | observed maximum |
| --- | ---: | ---: |
| DOF step | 0.061000 rad | 0.317250 rad |
| root translation step | 0.100000 m | 0.206091 m |
| stance slip p95 | Stage100 control ≤0.5445 m/s | 1.203336 m/s |
| root acceleration p95 | Stage100 control ≤14.9559 m/s² | 119.7878 m/s² |
| double-support inter-foot speed p95 | 0.05 m/s | up to 1.0295 m/s |

The largest jumps occur at direct support-side switches (for example B frame
85 and D frame 89).  Consecutive frames demand the COM be inside two laterally
separated sole rectangles.  The static frame optimizer can satisfy each side
individually, but not with the required temporal continuity.

## Conclusion

The failure is not “insufficient optimizer iterations” and not a left/right
mapping bug.  It is a modeling conflict:

1. the contact schedule switches support side without a usable load-transfer
   interval;
2. a hard static-COM-in-sole constraint changes its target discontinuously;
3. dynamic walking does not require COM projection to remain inside the stance
   sole at every instant—velocity, angular momentum, COP/ZMP and foot placement
   matter.

Therefore this per-frame KDMR-lite result is evidence, not a training cache.
The generated diagnostic directory contains `REJECTED_DO_NOT_TRAIN.md`.

## Next step

Do not loosen the continuity gate and do not relabel incompatible moving feet
as double support.  The next reference adapter should be trajectory-level:

1. jointly optimize a window/whole clip instead of independent frames;
2. treat contact timing or load transfer as a decision variable;
3. use COM velocity plus centroidal/ZMP/COP/GRF constraints (or a simplified
   capture-point/DCM surrogate), not static COM position alone;
4. keep upper-body/swing-foot semantics and exact mirror pairing;
5. if that is too heavy, retain Stage100-v2 and let the controller/policy use a
   bounded pelvis/stance-leg residual rather than baking discontinuous poses
   into the reference.

This remains a kinematic study; no claim about inverse dynamics, friction,
torque feasibility or real-robot safety is made.

## Evidence

- `docs/reports/x2_stage102_com_pose_kdmr_lite_retry_v2.{json,md}`
- `docs/reports/x2_stage102_com_pose_kdmr_lite_kinematic_gate.{json,md}`
- `docs/reports/x2_stage102_com_pose_kdmr_lite_feasibility.{json,md}`
- `tools/repair_x2_reference_com_pose_kdmr_lite.py`
- `tests/test_x2_reference_com_pose_kdmr_lite.py`

