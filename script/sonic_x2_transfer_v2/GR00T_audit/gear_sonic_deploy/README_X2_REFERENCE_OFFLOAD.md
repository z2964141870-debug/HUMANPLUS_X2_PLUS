# X2 3588S Reference Offload

Updated: 2026-08-30 CST

## 2026-08-30 checkpoint

The timestamp-corrected live dry-run completed successfully in
`logs/garment_live_dryrun_20260830_131441`:

```text
source:                 28.35 Hz, p95/max gap 43.6/71.5 ms
timestamp HMCP:         1140/1140, fallback/missing/reset 0/0/0
Sonic:                  1601 ticks, 50 Hz, HAL publishers disabled
fixed/timestamp speed:  1.727x median, 2.181x p95
policy clipped ticks:   55/1500 (previous live run: 359/1500)
```

This closed the live velocity-semantics issue. The subsequent pipeline-enabled
run below also closed the 30 Hz source-throughput gate.

A 1227-frame diagnostic capture on the 3588S split the active reference
calculation as follows:

```text
stage                  median / p95
physics                 15.46 / 19.47 ms
Fast-SMPL                2.49 /  3.76 ms
native GMR               6.75 / 10.45 ms
remote total            25.06 / 30.32 ms
```

The official 3588S `aima-agent-app` was also present during the successful
2026-08-29 run, so it was not a new regression. It nevertheless consumed
about two full CPU cores at nice `-19`. `aima em stop-app agent` stopped it
once but the platform restarted it after about eight seconds. For the current
user-authorized benchmark window, exact PID `13874` was suspended with
`SIGSTOP`; no MC, HAL sensor, interaction, or HDS process was stopped. This is
a temporary test condition, not yet a production startup procedure.

### Bounded one-frame pipeline

The client can now overlap local preprocessing/LFP for frame N+1 with the one
ordered 3588S RPC for frame N. The server still sees exactly one stateful,
monotonic sequence and at most one request in flight. There is no request or
result backlog, no local fallback, and a BLE pause discards the old in-flight
result before publishing resumes. A socket timeout is raised before another
frame can be submitted.

The A3 600-frame known-good raw replay passed on 2026-08-30 with the official
agent suspended:

| Mode | Wall rate | Remote median / p95 | qpos error vs old synchronous baseline |
| --- | ---: | ---: | ---: |
| Synchronous | 41.00 Hz | 16.16 / 17.70 ms | 0 rad |
| One-frame pipeline | 59.48 Hz | 14.87 / 16.72 ms | 0 rad |

The first live pipeline run then passed in
`logs/garment_live_dryrun_20260830_135137`:

```text
source:                  35.02 Hz, p95/max gap 30.5/46.6 ms
pipeline/server frames:  1497, discarded 0
remote processing:       16.78/18.01 ms median/p95, 19.42 ms max
timestamp HMCP:          1377/1377, fallback/missing/reset 0/0/0
Sonic:                   1601 ticks, 50 Hz
policy clipped ticks:    38/1500, maximum pre-clip 20.90
custom HAL publishers:   0; official MC remained active
```

All eight protocol/pipeline tests pass on both the Mac and SoC1, covering
ordering, one-frame delay, completed and in-flight discard, timeout
propagation, startup timeout, CRC rejection, and wire round trips. Live
throughput is now proven.

The deliberate remote-loss test then passed in
`logs/garment_live_dryrun_20260830_140817`:

```text
3588S:                  intentional disconnect at processed frame 500
server processing:      16.14/17.44 ms median/p95, 19.72 ms max
SoC1 source before loss: 35.03 Hz, p95/max gap 30.2/45.8 ms
HMCP:                   381 accepted/sent, then no further publication
Sonic:                  CONTROL -> SAFE_IDLE about 0.518 s after final HMCP
fallback/backlog:       none
custom HAL publishers:  0; official MC remained active
```

The 3588S log records `sequence_without_result=499`, so the in-flight result
was not delivered. The client surfaced EOF before another request, and the
garment worker stopped rather than falling back locally. This closes the final
unpowered offload/starvation gate. These results do not authorize powered HAL
output.

## 2026-08-30 diagnosed live-input anomaly

The first timestamp-mode live retry did not reach HMCP/Sonic measurement. BLE,
T-Pose, CUDA LFP, and the remote RESET passed, but the 3588S reference stage
slowed to `144.61 ms` median / `323.67 ms` p95 / `770.25 ms` maximum over 51
frames and then exceeded the steady-state 80 ms timeout. No HAL publisher was
created.

At the same time, the 3588S official `aima-agent-app` used about `182-188%`
CPU and the sensor process about `68-71%`; both ran at nice `-19` across all
eight cores. A fixed A3 raw replay under that same load remained much faster:
remote `26.46/34.46 ms` median/p95, RPC RTT `28.50/36.18 ms`, CUDA LFP
`3.87/5.56 ms`, and `27.02 Hz` wall throughput. Therefore general network or
model failure does not explain the live result. The subsequent bounded capture
split the exact live pose/velocity path into physics, Fast-SMPL, and native
GMR; its measured result is recorded in the checkpoint above.

## Status

The offload route is numerically exact on fixed replay, fast enough at live
rate, and bounded under deliberate server loss. It has not been used for
powered robot control.

The first live attempt on 2026-08-29 reached calibration and acknowledged the
remote RESET, then timed out during startup. A retry proved frame zero returned
but the next frame still took 123-140 ms, exceeding the 80 ms steady-state
deadline. The client now allows 1000 ms for the first 50 results and restores
the 80 ms deadline after that bounded warmup. This does not change the
disconnect/no-fallback behavior.

```text
SoC1: BLE -> TIC/LFP CUDA -> pose + velocity
                         -> TCP reference RPC
3588S: physics postprocess -> Fast-SMPL -> native GMR -> qpos36
                         -> TCP result
SoC1: HMCP -> v5.1 adapter -> Sonic 50 Hz -> dry-run
```

Sonic, robot feedback, safety gates, and the HAL writer stay on SoC1. Only the
stateful human-reference generator moves to the 3588S.

## Evidence

All tests below were isolated from MC and HAL.

| Test | SoC1 | 3588S A76 cpu6 | Result |
| --- | ---: | ---: | --- |
| Physics only, median / p95 | 12.94 / 13.82 ms | 14.74 / 21.52 ms | Identical output |
| Physics + Fast-SMPL + native GMR, median / p95 | 22.37 / 23.97 ms | 24.05 / 29.23 ms | Identical output |
| SoC1-to-3588S synchronous RPC, 400 frames | n/a | 27.13 / 33.62 ms RTT | 36.09 Hz wall rate |
| A3 known-good raw, 600 frames | CUDA LFP 4.04 / 5.91 ms | reference 23.61 / 29.32 ms | 29.82 Hz wall rate |

The RPC run reported:

```text
remote processing: 26.00 ms median, 32.36 ms p95
RPC-only overhead:  0.97 ms median,  1.54 ms p95
qpos36 max error:    0.0 rad
wall throughput:    36.09 Hz
```

The SoC1 and 3588S unified reference outputs had the same SHA-256:

```text
9c83cf92a80d36bd85f2d13ed0c27bb80e0e2ea1ae099934148c35f76ce2fb1e
```

The comparison also passed with exact zero error for input pose, input
velocity, physics pose, root translation, and final qpos36.

The A3 known-good replay used raw dataset SHA-256
`4bd785853e7dc0b5644871b17f4e7b3a179e72d8dc5aea130313106d24d82e70`.
It replayed the verified T-Pose, neutral, and early arms sequence through the
same SoC1 CUDA TIC/LFP and the live 3588S RPC. All 600 qpos36 outputs were
finite. Remote processing was 23.61 ms median / 29.32 ms p95 and the first 20
frames stayed below 56.8 ms. No BLE, Sonic, ROS, MC, or HAL publisher ran.

## Live Garment Dry-Run

The bounded 50-frame startup window allowed the full live route to complete on
2026-08-29. The run accepted 1137/1137 HMCP frames with zero invalid,
duplicate, or out-of-order frames. Corrected C++ Sonic completed 1601 ticks at
50 Hz with HAL publishers disabled and official MC still active.

```text
live source:       28.45 Hz
source gap p95:    43.0 ms
source gap max:    91.0 ms
SoC1 preprocess:   approximately 1.0-1.2 ms
SoC1 CUDA LFP:     approximately 4.2-5.1 ms
3588S processing:  approximately 24-27 ms
RPC RTT:           approximately 25-28 ms
```

This passes transport and dry-run correctness, but remains below the 30 Hz
powered-source gate. The dry-run policy also clipped raw actions on 359 of 1500
CONTROL ticks, with a maximum pre-clip magnitude of 21.74 against the trained
clip value of 20.0. That is a separate powered-control audit item; throughput
success must not be treated as powered readiness.

### Live reference velocity correction

The 28.45 Hz run still used the MuJoCo-parity derivative
`(q[t] - q[t-1]) * 50`. At that measured rate it represented approximately
1.76 times the timestamp-derived physical velocity and may have contributed to
the clipped policy inputs/actions. The adapter now has two explicit modes:

```text
fixed:     (q[t] - q[t-1]) * 50       # retained for MuJoCo parity/replay
timestamp: (q[t] - q[t-1]) / real_dt  # selected by both live garment launchers
```

HMCP's eight-byte header field is a little-endian float64 timestamp. The
garment sender writes `time.time()` immediately before UDP transmission. The
adapter parses that value, records it alongside receive-monotonic time, and
uses it for live differentiation. The first frame and intervals outside
`[5 ms, 200 ms]` produce zero velocity and establish a new derivative baseline;
the pose itself is still forwarded. This prevents clock rollback, burst, or a
long source gap from becoming a velocity spike.

Local regression evidence: all 13 HMCP adapter/summary tests and the original v5.1
byte-contract verifier pass. The 2026-08-30 live dry-run then confirmed zero
fallback, missing timestamp, and derivative-reset frames; clipped policy ticks
fell from 359 to 55 of 1500. This remains dry-run evidence, not powered-test
evidence.

Replaying the previous 1137-frame `21:57:19` live capture through the new
timestamp derivative found no invalid intervals. Median / p95 / maximum `dt`
were `34.49 / 43.01 / 91.00 ms`, with zero non-monotonic, too-short, or
over-200 ms gaps. Fixed-50 velocity was larger than timestamp velocity by
`1.725x` median and `2.151x` p95. Across all joints, absolute velocity p95 fell
from `1.404` to `0.814 rad/s`, and the maximum fell from `6.761` to
`3.975 rad/s`. These are adapter-input measurements, not policy-action results.

An affinity A/B replayed the same 600 known-good raw frames. CPU6 reached
29.82 Hz wall rate while CPU7 reached 29.27 Hz under the earlier official
load, so CPU6 remains selected. The bounded one-frame pipeline described above
now overlaps SoC1 LFP with the synchronous 3588S reference calculation and has
passed exact replay comparison. Live throughput subsequently passed at
35.02 Hz, and deliberate server loss stopped HMCP and tripped `SAFE_IDLE` in
about 0.518 s. The 30 Hz gate was not lowered.

## Protocol

The RPC is fixed binary data, not pickle or ad-hoc JSON. Every frame carries:

- protocol magic and version;
- message type;
- calibration epoch;
- monotonic sequence number;
- original monotonic timestamp;
- payload length and CRC32.

One request contains 24x3 axis-angle pose plus 24x3 velocity as little-endian
float32. One result contains 36 qpos values as little-endian float64 plus the
remote processing duration. The wire exchange remains synchronous, so there
is at most one in-flight frame and no old-frame queue. The optional local
worker only allows preparation of the next LFP frame while that exchange runs,
adding one source frame of output latency without changing server order.

An explicit RESET creates fresh physics and GMR state for each calibration
epoch. A timeout, disconnect, CRC error, epoch mismatch, or sequence mismatch
closes the client. The garment sender then stops producing HMCP. There is no
automatic fallback to local postprocessing because that would fork state.

Protocol unit tests cover frame/result round trips, CRC rejection, ordered
pipeline delay, stale-result discard, and timeout propagation.

## Candidate Files

Mac source:

```text
GR00T_audit/gear_sonic_deploy/scripts/offload/reference_offload_protocol.py
GR00T_audit/gear_sonic_deploy/scripts/offload/reference_offload_client.py
GR00T_audit/gear_sonic_deploy/scripts/offload/reference_offload_server.py
GR00T_audit/gear_sonic_deploy/scripts/offload/benchmark_physics_stage.py
GR00T_audit/gear_sonic_deploy/scripts/offload/benchmark_reference_pipeline.py
GR00T_audit/gear_sonic_deploy/scripts/offload/benchmark_reference_rpc.py
GR00T_audit/gear_sonic_deploy/scripts/offload/benchmark_raw_reference_rpc.py
GR00T_audit/gear_sonic_deploy/run_x2_garment_offload_dryrun.sh
/Users/yu/Documents/ChatGPT/X2/.remote_work/garment_udp_v2_reference_offload.py
```

SoC1 candidate paths:

```text
~/projects/smartwear_v2/garment_udp_v2_reference_offload.py
~/projects/x2_sonic_migrated_20260826/sonic_x2_transfer_v2/
  GR00T_audit/gear_sonic_deploy/run_x2_garment_offload_dryrun.sh
```

3588S isolated runtime:

```text
/agibot/data/home/agi/projects/x2_offload_bench_20260829
```

The original local sender and `run_x2_garment_live_dryrun.sh` were not changed.

Key candidate SHA-256 values after the timestamp and pipeline changes:

```text
172b9b6edc8f26688e8b531f7215dcce75c32aa1cd48a6e5582b98e6fbfb66e7  garment_udp_v2_reference_offload.py
a3cdfe8ed1d67ce8c9b8d7f626bcd372b564a969686be5f6dd8d17296a4cd96f  run_x2_garment_offload_dryrun.sh
60eaedeaca791613d773b6b8bd8a20b4e80cc5cdc96c3d58b996c8791fa94495  reference_offload_client.py
1a7c46de323599cb754c5e56bce74dd8831139d14697a55bd56e818f690cbf97  benchmark_raw_reference_rpc.py
bc09243d65c94117b6ba90b71cf9ca4522aa1fb037c3218adcba94fa2a8349ab  publish_v51_reference.py
99ba8ee1e04a3521301fa6b5b7e7da6ccdd35808dfad00e9eea1058f2e0062ed  summarize_live_dryrun.py
5c921f706cd245529aeda9ecfedd73d8d04f41abf40ae8d38a0fc237692c2568  test_hmcp_adapter.py
28837f388105cc337372997892b49568fee978088c2d5e07d5da18ab6ff643e5  reference_offload_protocol.py
002c04ac7dab1f0ca14e9e82dde8dcc65a81a714f4d6fc8cbd7a12c04f850384  reference_offload_server.py
```

## Completed Live Dry-Run and Diagnostic

Do not start this while another custom X2 deploy process owns or may own HAL.
First verify that the previous supported-policy wrapper has completed its own
handoff and that the robot is back under official MC.

Start the one-client server on the 3588S from an SoC1 terminal:

```bash
ssh agi@10.0.1.42
cd /agibot/data/home/agi/projects/x2_offload_bench_20260829
taskset -c 6 env \
  PYTHONPATH=runtime_g1_gmr_fast_20260829:. \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u reference_offload_server.py \
  --host 10.0.1.42 --port 51237 --threads 1 --max-iter 2 \
  --socket-timeout 2 --once
```

In another SoC1 terminal, run:

```bash
cd ~/projects/x2_sonic_migrated_20260826/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy
./run_x2_garment_offload_dryrun.sh
```

The operator enters `TPOSE` only when that terminal prints `T-POSE READY`.
This launcher still constructs no custom HAL publisher and leaves official MC
active. `X2_REFERENCE_PIPELINE=1` is now its default; set it to `0` only for a
deliberate synchronous A/B rollback.

Acceptance for this run:

- at least 30 Hz measured HMCP, with 33-35 Hz desired;
- zero invalid, duplicate, or out-of-order HMCP frames;
- corrected C++ Sonic remains at 50 Hz;
- no custom HAL command publisher;
- reference RPC timeout or server loss stops HMCP and trips the existing
  downstream watchdog rather than synthesizing frames.

All of these unpowered acceptance conditions passed on 2026-08-30. The
deliberate-loss variant used a server started with
`--disconnect-after-frames 500` and this SoC1 command:

```bash
X2_EXPECT_REFERENCE_DISCONNECT=1 ./run_x2_garment_offload_dryrun.sh
```

The next phase is not another transport optimization. First reproduce the
existing fixed-reference `neutral_damped` supported powered entry. Only then
connect this live source for one short, stationary-wearer, gantry-supported
interval; small live motion remains a separate later gate.
