# Baseline: suspended_sonic_20260828_170450

Status: passed for **gantry-supported neutral standing only**

## Identity

```text
launcher: gear_sonic_deploy/run_x2_supported_neutral_damped.sh
remote log: runtime_suspended/logs/suspended_sonic_20260828_170450
local analysis copy: gear_sonic_deploy/analysis_logs/suspended_sonic_20260828_170450
firmware: test-lx2501_3_t2d5-soc1-dc-v0.9.0-rc7
aimdk_msgs: 0.8.18
policy: x2_sonic_frozen_g1core_lora_v2.onnx
reference: StandStill / trained neutral default (no motion file)
policy rate: 50 Hz
HAL writer: 250 Hz
IMU: reconstructed pelvis
```

## Immutable manifest

| Artifact | SHA-256 |
| --- | --- |
| `run_x2_suspended_sonic.sh` | `29ad22cfc8a36ae8834eabd901ec52561ea134e12a3fdf42a8b8215fa1af705e` |
| `run_x2_supported_neutral_damped.sh` | `e551e3a360969c946b35541a730dc0eeb1b2de1106d01c991647f34148788b38` |
| `x2_deploy_onnx_ref.cpp` | `e33ab5fa78f7c89c7cb9ee3be24598c6b5e2a6ed91aee9fe884d71632de38d8d` |
| ONNX model | `8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9` |
| `x2_idle_stand.x2m2` (present but unused by this preset) | `d330f335d5bf1406b80cfaa5f53bf322c69f831a272aceb2d9320c8db5b8e7e8` |
| compiled `x2_deploy_onnx_ref` on SoC1 | `b750b7d8c78cdee42e3b0927c7bfe8cb7a6636b4e39a11b20da03cd4f058131c` |

The deployment copy is not a Git repository. The source checkout was dirty at
the time of this run; the artifact hashes, not `HEAD` alone, define the
baseline.

## Fixed preset

The relevant effective preset is:

```text
writer_hz=250
target_lpf_hz=8.0
target_lpf_hz_waist=2.5
kp_scale_ankle=1.5
kd_scale_ankle_pitch=3.31
kd_scale_ankle_roll=2.20
kd_scale_waist_pitch=3.0
policy ramp=4.0 s
policy return=2.0 s
target slew=0.12 rad/s
policy anchor=trained default
leg/waist/arm/head envelope=0.60/0.20/0.25/0.08 rad
relative tilt return=5 deg
absolute tilt return=25 deg
joint velocity return=0.8 rad/s
```

## Timeline and measurements

CSV phase timing:

| Phase | Start (monotonic s) | End (monotonic s) | Samples |
| --- | ---: | ---: | ---: |
| suspended reference pose ramp/static wait | 3094.120875 | 3122.759754 | 1433 |
| supported policy | 3122.779743 | 3422.748028 | 15000 |
| policy return | 3422.788020 | 3424.787934 | 101 |
| post-policy static hold | 3424.807915 | 4792.474387 | 68387 |

Policy duration was 299.968 seconds. Entry pelvis orientation was approximately
`roll +0.406 deg / pitch +7.003 deg`. The corresponding projected-gravity
vector was approximately `[+0.12193, -0.00703, -0.99251]`.

Entry joint positions included:

| Joint | Measured | Initial static target |
| --- | ---: | ---: |
| left ankle pitch | -0.4795 rad | -0.3630 rad |
| right ankle pitch | -0.4810 rad | -0.3630 rad |
| waist pitch | +0.3320 rad | 0.0000 rad |
| waist roll | +0.1578 rad | 0.0000 rad |

During the policy interval, reconstructed pelvis tilt remained approximately
4.13 to 7.08 degrees and ended near 4.46 degrees. In the first 30 seconds,
measured ankle velocity stayed at or below approximately 0.153 rad/s. The
gantry remained attached and load-bearing during policy settling; weight was
transferred only gradually. The exact gantry force and foot contact geometry
were not instrumented.

## Meaning of pass

This run proves that the frozen v0.9/0.8.18/SoC1/direct-HAL/250-Hz path can
produce a bounded neutral response under this supported entry condition for
five minutes. It also proves the handoff and cleanup chain can remain healthy.

It does not prove unsupported standing, walking, push recovery, arbitrary
entry posture, or garment teleoperation. It is one successful physical entry,
not a distribution of safe initial conditions.

## Known defects

- The posture was not upright.
- Waist pitch and roll had large command-to-measurement mismatch under load.
- Earlier visual observations included fore-aft sway and toe-rise tendency.
- Gantry force, foot pressure, and contact placement were not logged.
- The initial physical setup cannot yet be reproduced from software fields
  alone.

