# X2 Sonic S5-S7 Offline Integration Preparation

Updated: 2026-08-31 CST

## Scope and hard stop

This document is an offline dependency audit and a future acceptance card. It
does not authorize a robot session.

**S5 is not complete. All downstream powered work remains blocked by S5.** In
particular, do not start the garment source, reference offload, live-reference
proxy, or ARM request for a robot workflow; do not send MC/HAL commands; and do
not enter a powered garment policy, blend, hold, or S7 motion test.

The fixed-standing/support-transfer evidence needed by S5 is owned by another
thread. This document neither repeats that work nor changes any standing
controller, launcher, C++ deploy source, gate header, or C++ test.

## Stage snapshot

| Stage | Offline status on 2026-08-31 | Meaning |
| --- | --- | --- |
| S4: policy-off `x2_debug` | **Complete** | The exact M1 candidate publishes fresh measured debug during `GROUND_LOAD_HOLD`; the isolated dry-run process test reached `STANDSTILL_READY` without a HAL publisher. |
| S5: final parent and approval | **Blocked** | There is no valid same-candidate 300-second acceptance and no approved manifest. The 2026-08-31 attempt is invalid for fixed-support acceptance. |
| S6: stationary wearer, supported powered blend | **Blocked by S5** | No powered source/proxy/ARM/blend step is authorized. The current live launcher contract also has unresolved offline gaps listed below. |
| S7: small-motion acceptance | **Blocked by S5 and S6** | It requires a separate scope, mode, and acceptance series after S6 passes. |

## S4 evidence that is already closed

S4 is recorded as `M1_PREPOLICY_DEBUG_READY`, not as powered garment or
balance evidence.

- Exact deploy candidate SHA-256:
  `3c4dee77b9bde50c304b28200eea030b900ac55f9efc1473543068dbc9c3559f`.
- The isolated SoC1 candidate passed its two direct CTests.
- Proxy and parent-manifest suites passed `18/18` locally and on SoC1; the
  v5.1 byte self-check also passed.
- In the isolated process test, `--dry-run` disabled HAL command publishers and
  official MC remained the sole command owner.
- After simulated `load`, the proxy traversed
  `WAIT_ROBOT -> WARMUP -> STANDSTILL_READY`.
- At readiness, robot debug age was `0.008710 s`, guarded StandStill position
  error was at most `2.9564e-08 rad`, reference velocity was zero, and the root
  quaternion norm was `1.0`.
- A static policy-off control tick was allowed to remain zero while the debug
  ROS timestamp advanced by about `0.019989 s`.
- Source loss and candidate exit produced terminal LOCKOUT at source/debug ages
  of about `0.153/0.102 s` respectively.
- Cleanup left ports `5555/5556/5557` free and no custom control process.

Evidence directory:
`runtime_suspended/logs/m1_ground_debug_dryrun_20260831_104650/`.

## Local offline verification in this audit

Only the existing local proxy/wire/manifest tests were run. No source, offload,
proxy daemon, ROS process, C++ deploy, MC/HAL command, or robot connection was
started.

| Command, run from `scripts/garment_zmq` | Result |
| --- | --- |
| `python3 -m unittest -v test_gate_live_reference.py` | PASS, `12/12` |
| `python3 -m unittest -v test_validate_fixed_parent_manifest.py` | PASS, `6/6` |
| `python3 verify_v51_bytes.py` | PASS, all byte-contract checks |

The `18/18` unit results cover powered-debug rejection, T-Pose rejection,
continuous stillness, exact anchored StandStill, yaw rebase, blend endpoints,
source/debug staleness, terminal LOCKOUT, future-window rejection, artifact
hash mismatch, manifest evidence count, and powered proxy-status validation.

These results prove the tested pure logic only. They do not prove a valid S5
parent, a running-process-to-manifest identity binding, a ten-second powered
blend, or any positive S7 motion case. The byte self-check uses the retained
fixed-50-Hz parity fixture; the live source launcher uses timestamp-derived
velocity. Historical dry-run evidence closes the timestamp path, but this
self-check alone is not motion evidence.

## S5: evidence and approval still missing

### Current invalid attempt

The M1 candidate entered the final-parent profile on 2026-08-31 and remained
bounded for more than 170 seconds while the original heavy support stayed
fixed. The operator then loosened the sling at about policy second 174. That
changed the external support wrench during policy, so the run is **invalid** as
the required fixed-support 300-second acceptance and must not appear as a
passing manifest item.

Before the support change, median left ankle-pitch and waist-pitch tracking
errors were about `0.49/0.42 rad`. After the change, left/right ankle-pitch
errors reached about `0.686/0.615 rad`, policy tilt reached `13.34 deg`, and
waist-pitch speed reached `0.859 rad/s`; the joint-speed guard returned to
static PD. This is useful support-transfer evidence, not an S5 pass.

### Required S5 closure

S5 remains open until all of the following are present and independently
reviewed:

1. A valid final-parent acceptance using the exact M1 deploy candidate, frozen
   model, powered launcher, profile launcher, 50/250-Hz rates, reconstructed
   pelvis, `neutral_damped`, and fixed StandStill contract.
2. One accepted 300-second result whose support/contact condition does not
   change during the claimed fixed-support interval, with raw telemetry,
   physical support description, video, stop/return timeline, and final
   publisher ownership.
3. A deliberate approval scope from the standing/support-transfer owner:
   either the controlled marked support-transfer and staged release gates pass,
   or approval is explicitly restricted to a continuously taut load-bearing
   gantry. The limitation must be written in the approval note.
4. Final deploy binary, ONNX model, powered launcher, and profile launcher
   SHA-256 values, checked against the exact target-side files.
5. Three accepted short supported runs and the accepted 300-second run, with
   target-side log and summary paths plus summary SHA-256 values.
6. A real approval identity, timestamp, scope, and note supplied by the
   responsible human. A non-empty string is not proof of identity; do not
   synthesize or guess `approved_by`.
7. A real `approved_manifests/x2_fixed_standstill_parent.json` that passes the
   validator against the deployment root. This audit does not create it.

The current candidate identities recorded by the handoff are:

| Artifact | SHA-256 |
| --- | --- |
| M1 deploy binary | `3c4dee77b9bde50c304b28200eea030b900ac55f9efc1473543068dbc9c3559f` |
| ONNX model | `8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9` |
| `run_x2_suspended_sonic.sh` used by M2 | `7b6edcb1e23f2e7507335622ae3f7c57dce0754ddeba289d22bb17582a268ff6` |
| `run_x2_supported_final_parent_telemetry300.sh` | `1b73418811619fd701ff3fa3195dfe3ea0e40edf8f755b42b0d515ae92b184d8` |

These are candidate records, not an approved manifest.

### S5 contract issues found offline

The following must be resolved explicitly before approval; the validator's
green result alone would not close them.

| Issue | Audit finding | Required evidence/decision |
| --- | --- | --- |
| Short-run duration semantics | The three accepted relative-entry reports contain 1500 samples but record `policy_duration_s` as `29.978826`, `29.978888`, and `29.978866 s`. The validator requires the manifest's `duration_s >= 30.0` and does not compare that value with the summary. | Define and document whether acceptance is 1500 scheduled ticks or at least 30.000 seconds of sampled span. Do not silently round a measured `29.979` value up to `30.0`. |
| Short-run identity | The handoff accepts the three existing distinct nominal-30-second runs, while the validator does not prove they used the final M1 binary or compare evidence identity with artifact identity. | The approver must record the accepted inheritance rationale and exact hashes; the validator cannot make that judgment. |
| Evidence authenticity/distinctness | The validator checks `result`, claimed duration, paths, and summary hashes. It does not prove runs are distinct, parse summary result/duration, or inspect support/ownership evidence. | Human review of raw logs, summaries, physical record, and final ownership remains mandatory. |
| Approval identity | The validator only requires non-placeholder, non-empty approval strings. | Approval must come from the actual responsible person; successful parsing is not authentication. |
| Manifest root versus model path | ARM validation fixes `--root` to `gear_sonic_deploy` and rejects resolved artifact paths outside it. The frozen model used by the final wrapper resolves to `../../models/x2_sonic_frozen_g1core_lora_v2.onnx`, outside that root; no in-root model file exists locally. | Freeze a layout/validation contract that hashes the exact model actually loaded. An unrelated in-root copy is not sufficient unless the launcher is also bound to that same file. |
| Session binding | ARM validates files on disk and proxy freshness, but proxy status contains no running C++ binary/model/launcher hash and no positive `SUPPORTED_POLICY` state. | The approved live parent must hash-gate its actual runtime, and the operator must prove the running process is that parent and already in fixed StandStill policy before ARM. |
| Manifest enforcement | The proxy's interactive stdin command `ARM_LIVE_REFERENCE` calls the state-machine ARM path directly and does not validate a manifest. `--allow-dry-run-debug` can also relax the powered-debug gate for offline tests. | The powered workflow must make the approved manifest path non-bypassable. Never use direct proxy stdin ARM or `--allow-dry-run-debug` in S6/S7. |

## S6: stationary wearer, ten-second blend

**Every step in this section remains blocked by S5.** It is a future run card,
not a command to execute now.

### Additional implementation dependencies

Even after S5 evidence arrives, the current tree is not yet an executable S6
entry:

1. `run_x2_supported_garment_zmq.sh` is intentionally hard-disabled with
   `exit 4`; it starts no process. No other enabled wrapper selects
   `--supported-neutral-garment-zmq`.
2. The internal garment mode currently falls through to a `5.0 s` supported
   policy timeout, while the proxy blend is `10.0 s`. It therefore cannot
   demonstrate a complete ten-second powered blend.
3. The internal garment mode also falls through to the historical absolute
   pelvis entry gate (`pitch +5..+9 deg`, `|roll| <=2 deg`). It does not select
   the final parent's relative-to-load entry gate or its ankle-pitch tracking
   check. The absolute positive-pitch gate was rejected after it forced an
   artificial upper-torso harness pose; this mode is not equivalent to the
   accepted parent contract.
4. `arm_x2_supported_garment_live.sh` starts no controller. It only validates
   the manifest/status and writes one session-bound ARM trigger.
5. The ARM validator requires fresh powered debug and
   `STANDSTILL_READY`, but it cannot tell policy-off `GROUND_LOAD_HOLD` from
   fixed StandStill `SUPPORTED_POLICY`. The runtime state must be proved from
   the approved parent's own log before ARM.
6. The proxy also exposes a direct interactive `ARM_LIVE_REFERENCE` command
   that bypasses manifest validation. That path is forbidden for powered use;
   the approved workflow must make bypass impossible rather than relying on an
   operator convention.
7. The source launcher enforces `>=30 Hz` at initial readiness, then only
   reports the live rate every five seconds. A later rate below 30 Hz is not an
   automatic proxy lockout as long as frames remain under the `0.15 s` stale
   limit. Continuous throughput must therefore be an observed acceptance
   metric, not an assumed ARM gate.
8. Pre-ARM StandStill dwell time and post-blend stationary hold time are not
   frozen in the current contract. Both must be specified before a run; process
   lifetime or merely reaching `LIVE` is not a pass.

These gaps require a separately reviewed, hash-gated live parent contract.
This offline-prep task does not modify or promote a launcher.

### Complete future sequence

After S5 and the implementation dependencies above are closed, the S6 order is
exactly:

1. **Create a fresh session.** Treat battery change/reboot as invalidating all
   old calibration epochs, proxy sessions, status files, ARM requests, debug
   frames, and process IDs. Begin fully suspended with the physical emergency
   stop held and complete the repository's read-only ownership preflight.
2. **Verify the approved identity.** Check the manifest, target artifact
   hashes, approved support scope, empty/stale-free ports, absence of old
   garment/offload/proxy/deploy processes, and official MC as the only command
   publisher. Any mismatch blocks the session.
3. **Start the reference server, then the source-only path.** After the 3588S
   reference server is ready, start the SoC1 source. Wait for both garments and
   enter `TPOSE` only after the terminal prints `T-POSE READY`. Require the
   source to reach at least `30 Hz`; raw timestamp v5.1 pose is confined to
   `:5555`.
4. **Start a fresh safety proxy.** Use raw `:5555`, robot debug `:5557`, and
   guarded output `:5556`, with powered debug required. Do not use
   `--allow-dry-run-debug`. The proxy initially remains in `WAIT_ROBOT` and
   publishes nothing.
5. **Start the approved proxy-fed parent from full suspension.** Its launcher
   must hash-gate the actual deploy, model, powered launcher, profile, rates,
   and ZMQ mode. It must prove command silence and unique HAL ownership before
   enabling the 250-Hz writer.
6. **Establish the approved support while policy is off.** After
   `READY_FOR_GROUND`, establish the documented foot contact and support, then
   enter `load`. Hold static PD until robot feedback and entry telemetry are
   stable. Once policy turns on, sling tension, robot height, foot placement,
   and contact may not change.
7. **Wait for anchored StandStill readiness.** With fresh raw source and
   powered policy-off debug, the proxy must traverse
   `WAIT_SOURCE/WAIT_ROBOT -> WARMUP -> STANDSTILL_READY`. Readiness requires at
   least 50 accepted frames and two continuous seconds of neutral stillness.
   Guarded `:5556` now contains exact trained StandStill, zero reference
   velocity, and robot-yaw anchoring; live garment motion is not yet visible to
   Sonic.
8. **Enter fixed StandStill policy.** The C++ live path must independently see
   at least 40 warmed body frames, reference age within `0.5 s`, at least
   `0.8 s` of continuous freshness, explicit velocity, fresh robot state, and
   the approved supported-entry gate. Only then may the responsible operator
   request `policy`. Hold the wearer and support fixed for the predeclared
   pre-ARM observation interval. The parent log must positively show
   `SUPPORTED_POLICY`; proxy readiness alone is insufficient.
9. **Request ARM exactly once.** Recheck that the proxy status is fresh,
   `STANDSTILL_READY`, `stationary_only=true`, `debug_dry_run=false`, and tied
   to the current live PID/session. The ARM helper validates the approved
   manifest, creates the trigger with exclusive creation, and waits for
   `BLEND`. If it reports `LIVE_REFERENCE_ARM_OUTCOME_UNKNOWN`, do not repeat
   the request; inspect the same session's proxy status. Never type the direct
   proxy stdin ARM command.
10. **Complete the ten-second blend with the wearer stationary.** ARM captures
    one constant `yaw_delta = robot_yaw - wearer_yaw`. For
    `u = elapsed/10 s`, the proxy uses
    `alpha = u^2 * (3 - 2u)`. Joint position is
    `StandStill + alpha * (live - StandStill)`; joint velocity is
    `alpha * live_velocity + (d alpha/dt) * (live - StandStill)`; the root
    quaternion uses normalized interpolation to the fixed-yaw-rebased live
    quaternion. All nine future slots repeat the same blended position,
    velocity, and quaternion. At alpha zero the bytes are anchored StandStill;
    at alpha one they are the yaw-rebased live stationary reference. The
    wearer performs no intentional motion during the blend.
11. **Hold stationary after `LIVE`.** Continue for the predeclared short hold
    without changing wearer pose, support, feet, or contact. Reaching `LIVE`
    alone is not acceptance.
12. **End through the bounded return.** Request `stop` while policy is active
    and wait for the configured two-second return to the captured static hold
    and confirmed `GROUND_LOAD_HOLD`. Keep supporting the robot. Fully suspend
    it and confirm both feet are clear before `lifted` ends the writer and
    restores official MC. Only after control ownership is clean should the
    proxy/source/offload session be ended. Verify ports, processes, all four
    command publisher counts, and state feedback.

### S6 pass requirements

S6 passes only if all of the following hold:

- the complete ten seconds occurred while the approved policy was powered, not
  merely while the proxy's alpha advanced;
- the wearer stayed stationary through blend and the frozen post-blend hold;
- proxy states followed the expected sequence with no LOCKOUT;
- raw and guarded references, `x2_debug`, Sonic, HAL, and transition logs are
  complete and time-aligned;
- live source throughput remained acceptable and no fallback/reset occurred;
- there was one and only one custom HAL writer;
- support/contact did not change after policy entry;
- no growing oscillation, toe rise, unexpected sound, tilt/speed/tracking
  growth, safety return, or operator concern occurred;
- normal `stop -> static PD -> fully suspended -> lifted -> official MC`
  cleanup completed and final ownership was verified.

### S6 stop conditions and response

| Phase | Stop/block condition | Required response |
| --- | --- | --- |
| Before start | S5 incomplete; missing/invalid manifest; hash, process, port, ownership, support-scope, or battery-session mismatch | Do not start any S6 process. Remain in read-only diagnosis. |
| Before `policy` | Source below the accepted rate, rejected/reset HMCP, proxy not exactly `STANDSTILL_READY`, dry-run debug, stale stream, entry gate not stable, or support/contact not reproducible | Withhold `policy`; end from the safe current state. |
| Before ARM | Parent not positively in stable fixed StandStill policy, proxy/status/manifest not fresh, ARM trigger already exists, or wearer not stationary | Do not ARM. A stale/consumed session must be replaced, not reused. |
| During policy/blend/hold | Growing oscillation, toe rise, unexpected sound, increasing tilt/joint speed/tracking error, any safety trip, support/contact change, or operator concern | Request `stop` immediately and allow the bounded return. Use the physical emergency stop for dangerous motion. |
| Proxy fault | Any post-warmup source/debug/wire/stationary violation causes terminal `LOCKOUT` and guarded output ceases | Request `stop` immediately. The C++ stale watchdog is a backup: at `0.5 s` it starts the bounded return. Never restart/re-arm within the same loaded session. |
| Reference/debug loss | Raw age over `0.15 s`, debug age over `0.10 s`, non-increasing timestamps/indices, invalid wire, or process exit | Treat as LOCKOUT/stop; returning data does not authorize recovery. |
| Hazardous motion | Software return is not fast or safe enough | Physical emergency stop takes priority; keep physical support. |
| After return | Robot remains partly loaded or feet remain in contact | Do not use `Ctrl-C`, kill, close SSH, stop the control terminal, or send `lifted`. Fully suspend first. |

LOCKOUT is terminal for that proxy process. A reboot or battery change requires
a new calibration epoch, proxy session, status file, ARM request, preflight,
and runtime identity check.

## S7: separate small-motion acceptance matrix

**S7 is blocked by S5 and by an accepted S6. No S7 powered action is currently
authorized.**

### Dependencies that must close before S7

1. S6 must pass with the same frozen support scope and complete evidence.
2. S7 needs a separate explicit approval scope. The current manifest scope is
   `supported-stationary-live-reference`; it does not approve motion.
3. The proxy currently hard-codes `stationary_only=true`, and the ARM validator
   requires that value. Although the post-ready source envelope can numerically
   admit some bounded movement, using that behavior as S7 would exceed the
   approved semantics.
4. Existing tests prove that excessive post-ready motion locks out. They do not
   include a positive LIVE small-motion trajectory, return-to-neutral case,
   per-class boundary test, or S7 session/scope test.
5. Exact wearer amplitudes, rates, action durations, post-action dwell, and
   robot telemetry margins must be frozen after S6 evidence is reviewed. The
   current proxy/C++ limits are fail-closed ceilings, not demonstrated S7
   command amplitudes or pass thresholds.

Every matrix row below is a separate session with a fresh proxy/ARM and the
full S6 stationary qualification first. Each first acceptance contains one
predeclared slow outbound-and-return cycle only. Do not improvise repetitions
or combine rows during a run.

| ID | Isolated motion class | Reference/contact constraints | Pass evidence | Immediate stop/exclusion |
| --- | --- | --- | --- | --- |
| S7-0 | Stationary requalification | Repeat approved S6 entry, 10-second blend, and stationary dwell; support and feet fixed | Same S6 pass evidence; establishes that this session starts from the accepted parent | Any S6 stop condition; no motion row may begin |
| S7-A | Left arm only, one frozen plane | Right arm, waist, legs, root roll/pitch, feet, and support remain at the accepted stationary reference | Guarded left-arm reference and robot response are smooth, bounded, and return to the frozen stationary band without growth elsewhere | Contralateral/body motion outside the frozen band, sway growth, toe rise, sound, or proxy/safety fault |
| S7-B | Right arm only, one frozen plane | Mirror S7-A; left arm and lower body remain stationary | Same evidence as S7-A, independently recorded | Same as S7-A |
| S7-C | Bilateral symmetric arms | Only after S7-A and S7-B pass; no asymmetric reach, torso lean, or lower-body command | Both arms track smoothly; pelvis/ankle/waist telemetry shows no new growing response; return-to-neutral dwell passes | Any asymmetric loading, torso compensation growth, or common stop condition |
| S7-D | Small torso yaw only | Feet, knees, hip height, root roll/pitch, and support fixed; no waist pitch/roll combination | Yaw reference/rebase remains continuous; no heading snap, lateral/fore-aft growth, or lower-body tracking growth; returns to stationary band | Pitch/roll or contact change, heading discontinuity, proxy root-rate gate, or common stop condition |
| S7-E | Small symmetric lower-body flex-and-return | Only after upper-body rows pass; both feet planted, no step, no unilateral load transfer, no deep squat, and no deliberate support change | Hip/knee references and measured response remain symmetric; foot contact and support stay fixed; pelvis/ankle/waist signals remain bounded and return to the stationary band | Toe rise, foot unloading, asymmetry, meaningful centre-of-mass transfer, support change, or common stop condition |
| S7-F | Final combined small full-body cycle | Combine only directions and amplitudes already accepted in S7-A through S7-E; no new degree of freedom | All constituent reference/robot signals remain within their accepted bands, no cumulative drift appears, and the final stationary dwell passes | Any unapproved combination, accumulated drift, contact change, or common stop condition |

Walking, stepping, single-leg loading, deep squat, large torso pitch/roll,
push recovery, unsupported standing, and obvious centre-of-mass transfer are
outside S7. Head joints remain at the trained defaults in the frozen garment
mapping, and finger control is a separate path; neither is silently included
in this matrix.

### Common S7 acceptance rule

A row is not a pass merely because no hard safety trip occurred. It requires:

- a frozen input waveform with recorded peak joint offsets, velocities, root
  roll/pitch/yaw, and duration;
- proxy `LIVE` throughout, fresh source/debug, no wire rejection, and no
  LOCKOUT;
- a single writer and no stale/tilt/joint-speed return;
- robot tilt, speed, tracking error, ankle/waist effort, and support/contact
  trends compared against that session's S7-0 baseline;
- no growing oscillation, toe rise, unexpected sound, visible foot unloading,
  support change, or operator concern;
- return to the predeclared stationary band and completion of the frozen
  post-action dwell;
- normal bounded return, full suspension, `lifted`, official-MC restoration,
  and final ownership verification;
- raw garment/HMCP, raw and guarded ZMQ, proxy status/transitions, robot debug,
  Sonic/HAL telemetry, video, support description, hashes, and final recovery
  state preserved for that single row.

## Existing numeric gates: ceilings, not S7 targets

| Layer | Existing value |
| --- | --- |
| Source startup | at least `30 Hz` after the source readiness window |
| Proxy source/debug staleness | `0.15 / 0.10 s` |
| Proxy warmup | at least 50 accepted frames and 2 continuous seconds |
| Proxy root envelope | roll/pitch `<=12 deg` absolute; angular speed `<=0.35 rad/s` |
| Proxy joint offset envelope | leg/waist/arm/head `0.60/0.20/0.65/0.08 rad` |
| Proxy reference velocity envelope | leg/waist/arm/head `0.20/0.20/0.25/0.15 rad/s` |
| Proxy blend | `10.0 s`, smoothstep, terminal LOCKOUT after a post-warmup fault |
| C++ ZMQ entry | 40 body frames, age `<=0.5 s`, continuous freshness `>=0.8 s`, explicit velocity |
| C++ supported command envelope | leg/waist/arm/head `0.60/0.20/0.25/0.08 rad` |
| C++ return gates | relative tilt `+5 deg`, absolute tilt `25 deg`, joint speed `0.8 rad/s` |
| C++ live starvation | `0.5 s` then the configured `2.0 s` return to captured static hold |

The proxy arm offset ceiling (`0.65 rad`) is wider than the supported command
arm envelope (`0.25 rad`). Staying inside the proxy does not prove that the
end-to-end controller is unclamped or acceptable. S6 evidence must establish
smaller operational bands before S7 amplitudes are approved.

## Final no-go statement

As of this audit:

- S4 is complete for the exact offline/debug candidate.
- S5 lacks valid final-parent evidence, deliberate support scope, a valid
  end-to-end artifact layout, real approval, and an approved manifest.
- S6 additionally lacks an enabled hash-gated proxy-fed powered parent whose
  entry gates match the accepted parent, whose policy duration can cover the
  complete ten-second blend and frozen dwell, and whose manifest gate cannot
  be bypassed through proxy stdin.
- S7 lacks S6 evidence, motion scope, a positive motion mode/test contract, and
  approved per-row amplitudes and telemetry bands.

Therefore **all S6/S7 powered steps remain blocked by S5**, including starting
the garment source/offload/proxy for a robot workflow, issuing ARM, entering a
live policy, performing the ten-second blend, or trying any small motion.
