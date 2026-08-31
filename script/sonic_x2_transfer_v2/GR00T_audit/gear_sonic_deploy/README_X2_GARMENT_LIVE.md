# X2 Onboard Garment-to-Sonic Integration

> Powered integration now sits behind the isolated safety proxy documented in
> `README_X2_GARMENT_POWERED_INTEGRATION.md`. The historical direct
> `run_x2_supported_garment_zmq.sh` entry is disabled. This document remains
> the source/offload/HMCP semantics record.

Updated: 2026-08-30 CST

## Current checkpoint (2026-08-30 14:12 CST)

The pipeline-enabled full live dry-run passed transport, source throughput,
and policy compute:

```text
log:                    logs/garment_live_dryrun_20260830_135137
source:                 35.02 Hz, p95/max gap 30.5/46.6 ms
timestamp HMCP:         1377/1377, fallback/missing/reset 0/0/0
Sonic:                  1601 ticks at 50 Hz
policy clipped ticks:   38/1500, maximum pre-clip 20.90
remote processing:      16.78/18.01 ms median/p95, 19.42 ms max
custom HAL publishers:  0
official MC:            active
```

The `>=30 Hz` live-source gate is now closed without lowering its threshold.
The bounded one-frame pipeline also passed the A3 600-frame fixed replay at
`59.48 Hz`, with exact zero qpos error against the old synchronous result. The
same no-agent condition produced `41.00 Hz` synchronously, so both the official
3588S agent load and serial LFP/RPC scheduling were measured separately.

The deliberate reference-loss test also passed:

```text
SoC1 log:               logs/garment_live_dryrun_20260830_140817
3588S server action:    intentional disconnect after processed frame 500
last HMCP timestamp:    1788070172.0570555 (381 accepted/sent)
Sonic transition:       CONTROL -> SAFE_IDLE at 1788070172.575439464
bounded stop time:      approximately 0.518 s after the final HMCP frame
fallback/reordering:    none
custom HAL publishers:  0; official MC remained active
```

The client propagated the server EOF, stopped HMCP production, and did not
submit another reference request or switch to a local stateful fallback. The
Sonic watchdog tripped at its configured `0.5 s` stale threshold and held the
dry-run default-angle safe state. This closes the final unpowered starvation
gate; it is not powered-control evidence.

## Objective

Run the complete V2 garment teleoperation path on the target X2 itself:

```text
V2 jacket/pants
-> X2 AX210 BLE
-> TIC4Clothes + LFP
-> Fast SMPL
-> GMR to G1 qpos36
-> localhost HMCP UDP :51234
-> G1-to-X2 mapping + LiveMotion
-> localhost ZMQ v5 pose :5556
-> Sonic policy 50 Hz
-> safety gates
-> HAL writer 250 Hz
-> the same X2 robot
```

There is no laptop or Mac in the runtime data path. The Mac is an SSH and log
console only.

## Current integration checkpoint (2026-08-29 21:25 CST)

The target-side transport and policy-compute chain has now run end to end:

```text
V2 BLE -> TIC/LFP -> Fast SMPL -> G1 GMR -> HMCP -> ZMQ v5.1
       -> corrected C++ tokenizer -> Sonic 50 Hz --dry-run       PASS

SoC1 source throughput target (>=30 Hz, desired 33-35 Hz)        NOT MET
Sonic HAL writer / powered garment tracking                      NOT TESTED
```

The first run above used the slow donor-compatible source path. The optimized
source candidate is installed without replacing that donor runtime:

```text
~/projects/smartwear_v2/runtime_g1_gmr_fast_20260829
```

The first optimized live run is:

```text
logs/garment_live_dryrun_20260829_200106
```

It accepted `943/943` HMCP frames and completed 1601 C++ ticks with no custom
HAL publisher. Source throughput increased from `13.85 Hz` to `23.04 Hz`
(`50.2 ms` p95 gap, `78.0 ms` maximum), but remained below the `30 Hz` gate.
Fast-SMPL dropped to about `3.8 ms`; native GMR ran at roughly `8-9 ms`. The
remaining dominant stage was the unchanged full-body physics postprocessor at
roughly `23-24 ms` per frame.

Offline SoC1 A/B then showed that this planner's tiny tensors run faster with a
single PyTorch CPU thread: `22.956 ms` median at the default 8 threads,
`19.478 ms` at 1 thread, and `21.228 ms` at 2 threads, with identical output.
The proven MuJoCo launcher also uses two GMR refinement iterations, not the
three used by the first onboard optimized run. The current formal launcher
therefore defaults to one CPU thread and `gmr_max_iter=2`; its SHA-256 on Mac
and SoC1 is
`7d84009d0a73258a5e3fb130b20f76b6b9211520a17b5ee129def27a57bbb5a2`.
The exact 23.04 Hz parent is retained on SoC1 as
`run_x2_garment_live_dryrun.sh.native_iter3_threads8_20260829`, and the original
baseline launcher remains `run_x2_garment_live_dryrun.sh.baseline_20260829`.

The promoted one-thread/two-iteration live run is:

```text
logs/garment_live_dryrun_20260829_201237
```

It accepted `950/950` HMCP frames at `23.25 Hz` (`48.3 ms` p95 gap,
`76.1 ms` maximum) and completed 1601 C++ ticks. This confirms that reducing
the GMR refinement count and CPU thread pool does not materially move the live
throughput ceiling. The local optimization route has therefore plateaued near
23 Hz.

An isolated 3588S reference-offload candidate has since passed exact numerical
comparison and a 400-frame TCP replay at `36.09 Hz` wall throughput. Its design,
benchmarks, files, failure behavior, and next dry-run are recorded in
`README_X2_REFERENCE_OFFLOAD.md`. The original local launcher remains the
rollback baseline.

The first complete offload live run reached `28.45 Hz` but exposed a separate
reference-velocity mismatch: the live adapter still multiplied each arriving
frame delta by the MuJoCo nominal `50 Hz`, overstating velocity by about `1.76x`
at that measured rate. The adapter now retains `fixed` mode for exact MuJoCo
parity and uses the HMCP sender timestamp in explicit `timestamp` mode for both
live garment launchers. Invalid, non-monotonic, shorter-than-5 ms, or
longer-than-200 ms intervals reset the derivative to zero. Local adapter and
wire-contract tests pass. The corrected 2026-08-30 live run measured zero
derivative resets and reduced clipped CONTROL ticks to 55/1500; the timestamp
fix is therefore complete.

Here `native GMR` means that target preprocessing is implemented by the ARM64
C++ library while the numerically sensitive IK remains in Mink/MuJoCo/DAQP. It
is the low-latency GMR route, not a second retargeting algorithm.

The first successful live session is:

```text
logs/garment_live_dryrun_20260829_192736
```

Measured evidence from that session:

- jacket `F7:C6:1F:AB:37:E0` and pants `DD:65:A4:4A:22:36` both
  connected with V2 137-byte frames;
- the operator explicitly entered `TPOSE`; the launcher then collected 595
  valid HMCP frames with zero invalid, duplicate, or out-of-order frames;
- HMCP ran at `13.85 Hz` (`71.1 ms` median gap, `83.6 ms` p95,
  `155.4 ms` maximum), not the required onboard target;
- corrected C++ Sonic completed 1601 logged ticks, including 30 seconds of
  CONTROL at 50 Hz and a two-second ramp-out;
- sampled reference age during CONTROL stayed at or below `81 ms`; no
  starvation transition occurred;
- the dry-run node constructed no HAL command publisher and official MC
  remained active;
- binary SHA-256 was
  `64d2663431f7531ce907bc006f6979ed34d05528ee335c3b71e8f4a48cc98bf6`;
  model SHA-256 was
  `8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9`.

The terminal's final `Aborted (core dumped)` came after both PASS messages
while the launcher interrupted the Python garment process. `garment.log`
shows `KeyboardInterrupt` during interpreter thread shutdown followed by
`terminate called without an active exception`. It did not invalidate the
captured data or C++ run. The launcher now terminates that native-threaded
process with `SIGTERM`, monitors both source children while waiting for manual
`TPOSE`, and reports measured HMCP timing at exit.

This result closes these integration questions on the target X2:

- live dual-BLE input, T-Pose calibration, and onboard garment inference;
- HMCP parsing and capture;
- named G1-to-X2 mapping at `pose-scale=0.7`;
- WXYZ-to-XYZW root conversion and v5.1 packing;
- strict C++ reference acceptance, tokenizer, and ONNX execution;
- 50 Hz policy scheduling and read-only HAL graph behavior.

It does not close powered behavior. During dry-run the real joints remain
under official MC and cannot follow policy targets, so the policy eventually
sees a counterfactual open loop. Forty-seven ticks exceeded the raw action
limit before clipping. The clipped CSV values were confined to right-wrist
roll and head yaw; head targets were bypassed to zero. Large late target
excursions in this run are therefore not a powered stability result, but they
also cannot be used as evidence that powered tracking is safe.

### Launcher

After starting the one-client reference server described in
`README_X2_REFERENCE_OFFLOAD.md`, power-cycle both garments once and run this
manually on SoC1:

```bash
cd ~/projects/x2_sonic_migrated_20260826/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy
./run_x2_garment_offload_dryrun.sh
```

The launcher waits for both garments, prints `T-POSE READY`, and accepts
`TPOSE` only from the operator's terminal. It never stops MC and cannot create
HAL command publishers. Logs are written automatically under
`logs/garment_live_dryrun_<timestamp>/`.

Default optimized source settings are:

```text
GMR root:     ~/projects/smartwear_v2/runtime_g1_gmr_fast_20260829
Fast-SMPL:   cpu
GMR backend: native
GMR iterations: 2
PyTorch CPU threads: 1
```

For diagnosis only, the old behavior can be selected without changing files by
setting `X2_GMR_BACKEND=baseline` and the corresponding donor
`X2_GMR_ROOT`. Do not use that rollback for a powered run.

### Next work, in order

1. Before any powered run, verify that no old custom PD/HAL owner remains and
   that the robot's support/contact state is explicitly known.
2. **Completed 2026-08-30:** the pipeline-enabled live run reached `35.02 Hz`
   with no rejected frames, 50 Hz Sonic, and no custom HAL publisher.
3. **Completed 2026-08-30:** the 3588S intentionally disconnected after frame
   500; HMCP stopped and Sonic entered `SAFE_IDLE` about `0.518 s` after the
   final frame, with no fallback and no HAL publisher.
4. Keep the powered garment gate closed until the fixed-reference
   `neutral_damped` Sonic entry is reproducible under support/contact. Then do
   one stationary-wearer supported interval before allowing small motion.

The earlier 300-second supported standing result proves that MC pause, HAL
ownership, custom PD, and fixed-reference Sonic can run. It does not prove
arbitrary supported entry, unsupported balance, or garment tracking. The
completed source pipeline and the remaining powered-balance gate are
independent; do not reopen the completed HMCP/ZMQ/C++ transport work when
balance fails.

## Core unanswered question

Porting the already-proven garment environment to another X2 is an engineering
prerequisite, not the main research risk. The unresolved sim-to-real question
is:

```text
Can the exact live LiveMotion reference drive the real Sonic observation,
policy, safety, and HAL loop without an interface mismatch or a physical
whole-body/contact instability?
```

This has two separate acceptance claims:

1. **Closed-loop integration:** live garment references reach the real Sonic
   tokenizer with the same semantics as the successful MuJoCo path, Sonic
   produces actions, and the guarded 250 Hz writer reaches the robot.
2. **Powered stability:** with that path active, the supported real X2 remains
   bounded first for a stationary wearer and then for small full-body motion.

The first claim is expected to be tractable because both endpoints already
exist, but it is not proven until the tokenizer inputs and actions match. The
second claim cannot be inferred from transport success or MuJoCo and requires
a bounded powered test.

## Machine roles

| Machine | Address | Role |
| --- | --- | --- |
| 5060 laptop | `humanplus@100.101.30.68` | Proven MuJoCo/shadow source environment and artifact donor; not part of the final runtime path |
| Target X2 SoC1 | `agi@192.168.43.21` | Final runtime: BLE, garment inference, reference conversion, Sonic, and HAL writer |
| Target X2 SoC0 | internal | Official MC and HAL host; official MC is paused only during the custom writer handoff |
| Target X2 3588S | `agi@10.0.1.42` from SoC1 | Candidate reference offload only: physics postprocess, Fast-SMPL, and native GMR |
| Mac | `192.168.43.33` on the robot LAN | SSH, deployment, monitoring, and log collection only |

Do not build an SSH or ZMQ relay from the 5060 laptop through the Mac to the
robot. That is the wrong topology.

## Proven source baseline

The following path has already run end to end in X2 official MuJoCo at about
33-35 Hz:

```text
AX210 V2 137-byte BLE frames
-> jacket 6 IMUs + pants 5 IMUs
-> T-Pose calibration
-> TIC4Clothes + LFP full-body reconstruction
-> 24-joint SMPL pose + root translation
-> Fast SMPL conversion
-> native/C++ GMR preprocessing and G1 retargeting
-> qpos36: root xyz[3] + root quaternion WXYZ[4] + G1 joints[29]
-> HMCP UDP :51234
-> G1-to-X2 31-DOF incremental mapping
-> LiveMotion
-> Sonic
-> X2 official MuJoCo
```

The validated launch parameters are:

- local LFP;
- Fast SMPL;
- native GMR preprocessing;
- garment target rate 35 Hz;
- `pose-scale=0.7`;
- no HP3090 PKL and no replacement planner.

Source entry points:

- `/Users/yu/Documents/ChatGPT/X2/.remote_work/run_v2_sim_shadow.sh`;
- `/Users/yu/Documents/ChatGPT/X2/.remote_work/garment_udp_v2_reconnect_safe.py`;
- `/Users/yu/projects/humanplus_x2_bridge/x2_realtime_closed_loop.py`.

This baseline proves real garment input through a closed MuJoCo Sonic loop. It
does not prove target-X2 dependency compatibility, onboard throughput, HAL
control, or powered garment tracking.

## Reference contract

### HMCP

The garment process publishes one G1 reference per valid reconstruction frame:

```text
20-byte HMCP header + 36 little-endian float32 qpos values
```

The receiver must reject bad magic, wrong qpos count, short packets,
non-finite values, and stale input. When BLE or reconstruction pauses, the
garment process already withholds HMCP; the downstream adapter must likewise
withhold ZMQ rather than synthesize a stand frame.

### G1 to X2

Keep the mapping in `x2_realtime_closed_loop.py` exactly:

```text
x2_joint = X2_default + 0.7 * (g1_joint - G1_default)
```

The mapping is by joint name, including the X2 waist/wrist ordering changes.
The two X2 head joints remain at the trained defaults.

### LiveMotion parity

The proven bridge uses a 600-frame rolling buffer, nominal `fps=50`, and a
40-frame warm-up. Sonic always samples the newest frame. At the live edge all
10 tokenizer reference indices clamp to that newest frame.

Therefore the v5 payload must repeat the newest joint pose and root
orientation in all nine strictly-future slots. A subtle but important detail:
the old tokenizer does not make the clamped future velocity zero. It uses:

```text
reference_joint_velocity = (latest - previous) * 50
```

for current and future slots. The onboard adapter/receiver must preserve this
in `fixed` mode before claiming MuJoCo parity. Live HMCP uses an explicit
timestamp-derived velocity instead; the C++ receiver consumes that field and
does not re-estimate it.

HMCP root quaternions are WXYZ; the current ZMQ wire is XYZW. Normalize before
publishing. Root translation is retained for diagnostics but is not consumed
by the current Sonic tokenizer.

The garment T-Pose frame and robot entry yaw must also be aligned once at
policy entry. Apply one constant yaw transform; do not rewrite the upstream
motion or add a predictor.

## Target X2 runtime

Current target identity and installed components:

- X2 SoC1 is NVIDIA Orin/aarch64 on firmware
  `test-lx2501_3_t2d5-soc1-dc-v0.9.0-rc7`;
- AX210 `hci0` works with both V2 garments;
- the `teleop` Python 3.10 environment provides the CUDA ONNX Runtime,
  NumPy, pyzmq, Bleak, and the garment dependencies;
- the retained donor G1 runtime lives under
  `~/projects/smartwear_v2/runtime_g1_gmr`, while the isolated optimized
  candidate lives under
  `~/projects/smartwear_v2/runtime_g1_gmr_fast_20260829`;
- the HMCP compatibility runtime lives under
  `~/projects/smartwear_v2/runtime_hmcp_compat`;
- the corrected C++ runtime lives under
  `runtime_suspended/ws/install_gatecheck_corrected_20260829`;
- after the 19:27 dry-run, all garment, adapter, and Sonic children exited,
  ports `51234`/`5556` were free, and official MC remained the only control
  path.

The software-installation, live-transport, throughput, and unpowered
starvation gates are closed. The remaining work is powered entry and physical
closed-loop behavior, not missing dependencies or reference transport.

## Current blockers

1. **The fixed-reference powered entry must be reproducible.** The earlier
   300-second supported result proves the route can work once, but arbitrary
   entry state, support/contact, and the frozen `neutral_damped` profile still
   need one repeatable bounded procedure before garments are connected to HAL.
2. **Real whole-body stability with a live reference is unknown.** A ZMQ frame
   reaching Sonic does not prove that the real robot can track it. The existing
   supported-neutral success/failure split means stationary live reference and
   small full-body live motion need separate measured intervals within the
   first bounded powered run.
3. **Dry-run is not a physical closed loop.** Official MC keeps the measured
   joints near Standing while Sonic's unapplied targets enter action history.
   Late dry-run drift is expected under that mismatch and cannot validate or
   invalidate real tracking. The next powered experiment must begin with a
   stationary wearer and full gantry support.
4. **The powered launcher is not promoted yet.** It must combine the corrected
   binary, `neutral_damped` gains/filter/writer settings, warmed/fresh ZMQ entry
   gates, bounded starvation return, and the existing supported-entry checks.
   Do not reuse an older `install_gatecheck` or change gains while wiring it.

## Acceptance gates

Complete these in order:

1. **Complete:** recorded HMCP adapter/wire tests, exact 680-D tokenizer and
   990-D proprioception parity, Python/C++ action parity within `3.099e-6`, and
   historical replay through the corrected C++ binary.
2. **Complete:** the live garment/offload/HMCP/ZMQ/Sonic dry-run reached
   `35.02 Hz`, accepted timestamped HMCP without fallback or reset, sustained
   50 Hz Sonic, and created no HAL publisher.
3. **Complete at final live rate:** intentional 3588S loss stopped HMCP and the
   C++ watchdog entered `SAFE_IDLE` about `0.518 s` after the final frame. Exit
   still requires fresh input plus the configured operator resume gate.
4. **Pending:** reproduce a bounded fixed-reference supported entry with the
   frozen `neutral_damped` profile.
5. **Pending:** in one short gantry-supported powered garment run, hold the
   wearer stationary for the first interval. This answers whether
   live-reference Sonic can maintain the same bounded neutral behavior.
6. **Pending:** only if the stationary interval is bounded, perform a second
   short interval of small full-body motion, including small lower-body
   reference changes.
   Stop at the first growing oscillation, toe rise, unexpected sound, stale
   reference, or joint-speed growth.

Do not send `policy` merely because a pose publisher exists. The launcher must
show a warmed and fresh v5 reference, stable robot feedback, and an acceptable
supported entry state.

## Explicit non-goals

- Do not use `walk_lower_garment_upper.pkl` or any HP3090 motion file.
- Do not introduce a second planner, a fixed gait, or direct garment-to-X2
  GMR.
- Do not move garment inference back to the 5060 laptop for final runtime.
- Do not put the Mac in the real-time data path.
- Do not change the 50 Hz policy rate, 250 Hz writer, model, gains, or pelvis
  reconstruction while closing the garment integration gates.
