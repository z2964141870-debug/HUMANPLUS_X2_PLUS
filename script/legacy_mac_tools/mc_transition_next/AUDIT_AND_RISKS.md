# Audit and risk register

## Existing v1.0 findings

The audit covered the current probe, hold, shell gates, tests, 2026-08-19
investigation, handoff safety note, and the 2026-08-25 hold log.  Existing live
hold and step wrappers remain blocked and were not changed.

1. `x2_v1_state_probe.py` intentionally enters `Ready` before `Develop_MC` and
   creates no HAL publisher.  That reproduces the previously identified torque
   gap rather than testing continuous ownership.  Its feedback audit counts
   receive gaps but has no source age, group cohort, sequence, or powered-command
   validity gate.
2. `x2_v1_hold_probe.py` improves overlap and handback substantially: it
   prepublishes the exact official command, keeps the mirror through `Ready`,
   requires fresh powered post-handback commands, and refuses release on an
   uncertain migration.  These invariants should be retained.
3. The hold probe uses local callback timestamps for freshness.  Old samples
   republished by an upstream node can look fresh, and four groups can represent
   different source times.
4. Command continuity is assumed from a 2 ms timer and publisher existence.
   There is no independent acknowledgment of all-group cycle emission, maximum
   publish gap, or downstream consumption.
5. The 2026-08-25 run reached `Develop_MC` and later verified powered official
   Standing, but the hardware still vibrated and felt softer.  The pass logic
   used position drift; a static whole-body PD controller can oscillate around
   its target with little drift.  Velocity RMS, acceleration RMS, reversal rate,
   current/torque saturation, IMU angular motion, and contact/balance metrics
   were not live trip inputs.
6. The candidate copied official Standing gains directly.  Earlier logic used
   35 percent gains.  Neither abrupt copy nor scalar reduction proves closed-loop
   stability under `Develop_MC`; gain changes must be part of a validated full
   Sonic PD contract and a continuous, jointly tested contract ramp.
7. Recovery is tied to the ROS process that also publishes the mirror.  If that
   process or context dies, Python cleanup cannot preserve torque.  An external
   supervisor and independent physical E-stop are required.
8. Normal handback changes the mirrored target/profile to live official output
   when it appears.  The post-marker/dwell checks are good, but the approach
   lacks a pre-Ready, bounded full-contract return ramp and an independent
   output continuity acknowledgment.
9. Arms-only Sonic with frozen lower body is already ruled out by the existing
   safety note.  The next controller must use the complete verified 31-joint
   Sonic contract or a vendor-supported upper-body overlay that leaves official
   balance ownership intact.

## Risk register

| ID | Risk | Consequence | Required control |
|---|---|---|---|
| R1 | State reports success before command ownership is effective | Torque gap or competing publishers | Publisher prewarm, downstream ownership evidence, bounded overlap |
| R2 | Locally fresh but source-stale/mixed-time feedback | Incorrect anchor and unstable PD error | Source age, sequence, group skew, stable dwell |
| R3 | Timer/process stall | Missing HAL cycles and sudden softening | Independent output monitor and real-time supervisor |
| R4 | Static PD oscillation with low position drift | Vibration, heating, loss of balance | Velocity/acceleration/reversal/current/IMU/contact trips |
| R5 | Abrupt or scalar gain change | Softening or high torque transient | Atomic C2 full-contract ramp validated at every blend point |
| R6 | Sonic packet gap/restart/replay | Target discontinuity | Session identity, strict sequence, local rate/accel/jerk bounds, latched stop |
| R7 | Fault path destroys publisher | Immediate torque loss | Post-intent exit lock; verified official overlap or physical E-stop only |
| R8 | Frozen command is mistaken for safe recovery | Balance controller stops evolving | Independent vendor-supported fallback controller |
| R9 | Official handback emits zero-gain `Business` commands | Mirror release leaves robot soft | Post-marker powered profile plus continuous dwell and output overlap |
| R10 | Competing or spoofed publishers | Nondeterministic actuator command | Publisher identity/ownership enforcement below ROS graph inspection |
| R11 | Whole-body joint/gain order mismatch | Wrong-joint torque | Hash and validate the complete 31-joint contract as one artifact |
| R12 | Laptop/ROS/Python is the only supervisor | No recovery after process/context loss | On-robot real-time supervisor and independent E-stop |

## Conditions that block all hardware testing

Do not run any real `Develop_MC`, MC state migration, or HAL publisher until
every condition below is satisfied.  Suspended support and an E-stop do not
waive these gates.

1. Vendor documentation confirms the exact firmware/AimDK handoff and recovery
   sequence, ownership semantics, required command rate/QoS, and supported
   overlapping-publisher behavior.
2. A complete 31-joint Sonic control contract is versioned and hashed: joint
   order, target transform, stiffness, damping, effort, velocity, torque limits,
   action scale, default pose, and update rate.  No official-gain/Sonic-target
   mixture is allowed.
3. Closed-loop simulation/replay passes takeover, every ramp blend point,
   candidate hold, active control, normal return, feedback loss, Sonic loss,
   output gaps, state-service timeout/rejection, process death, and handback.
4. Quantitative limits are justified from evidence for vibration, IMU angular
   velocity/tilt, contact, joint velocity/acceleration, tracking error, torque or
   current saturation, output gaps, and balance/fall precursors.
5. Source-stamped, sequenced feedback and official commands pass the freshness
   contract.  Local callback age alone is insufficient.
6. An independent monitor verifies all-group command continuity and downstream
   actuator consumption.  ROS graph publisher names are insufficient.
7. A supervisor outside the control process can keep or replace command
   ownership after Python/ROS context death.  Its behavior is fault-injected
   and verified.
8. A physically safe fallback controller is demonstrated for all post-takeover
   faults.  “Freeze last target” and static whole-body PD are not accepted as
   proof of balance recovery.
9. Official handback is proven to overlap with fresh, powered, post-marker
   Standing commands for the required dwell, without a stiffness drop.
10. Offline tests in this directory and the existing repository tests pass on
    the exact code/artifact versions proposed for integration.
11. A formal review resolves R1-R12 with named evidence, and a separate operator
    authorizes a short suspended test with physical exclusion zone and E-stop.
12. Mirror-only and transition tests use new, non-live adapters that preserve
    the hard locks in current scripts until the review explicitly approves a
    staged unlock.  The current true-robot scripts must not be directly edited.
