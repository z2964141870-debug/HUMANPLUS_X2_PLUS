// Pure-CPU unit test for the powered live-reference (garment -> ZMQ v5) gates.
// No ROS 2, no ONNX runtime, no hardware -- it exercises the exact decision
// functions the deploy binary calls, from supported_policy_gates.hpp:
//   - EvaluateZmqPolicyEntryGate : may a 'policy' request be accepted?
//   - EvaluateSupportedPoseEntryGate : is the measured pelvis entry stable?
//   - EvaluateSupportStepRequestGate : may one timestamped manual support
//                                      transfer step begin?
//   - ShouldPublishGroundLoadDebug : can policy-off robot telemetry be emitted
//                                    without masking stale HAL feedback?
//   - ZmqReferenceStale          : has the live reference starved during
//                                  SUPPORTED_POLICY (-> bounded 2 s return)?
//
// These guard a POWERED transition on a partly/fully supported robot, so the
// behaviours below are the ones that must not silently regress:
//   * policy is refused with no body reference;
//   * policy is refused before the warmup frame count;
//   * policy is refused until the reference has been CONTINUOUSLY fresh;
//   * strict-velocity mode refuses until an explicit-velocity frame arrived;
//   * a fresh, warmed, continuous, explicit-velocity reference is accepted;
//   * SUPPORTED_POLICY staleness (or a never-seen frame) trips the watchdog;
//   * a disabled watchdog never trips (the non-zmq / --disable path).
//
// Build standalone (no ROS 2 install needed):
//   cmake -S . -B build -DAGI_X2_OFFLINE_SYNTAX_CHECK=ON
//   cmake --build build
//   ctest --test-dir build --output-on-failure

#include "supported_policy_gates.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

#define EXPECT(cond, msg)                                                   \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " #cond     \
                << "  " << (msg) << "\n";                                   \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

using namespace agi_x2;

// A reference snapshot that PASSES every entry check. Each test below mutates
// exactly one field so a failure names the single offending condition.
static ZmqReadinessInputs ReadyInputs()
{
  ZmqReadinessInputs in;
  in.input_type_is_zmq      = true;
  in.has_body_reference     = true;
  in.body_frames_received   = 40;
  in.warmup_body_frames     = 40;
  in.reference_age_s        = 0.02;   // 50 Hz -> ~20 ms since last frame
  in.entry_max_age_s        = 0.5;
  in.continuous_fresh_s     = 1.0;    // fresh for 1 s
  in.min_continuous_fresh_s = 0.8;
  in.strict_velocity        = true;
  in.has_explicit_velocity  = true;
  return in;
}

static SupportedPoseEntryInputs ReadyPoseInputs()
{
  SupportedPoseEntryInputs in;
  in.state_fresh = true;
  in.pelvis_pitch_deg = 7.0;
  in.pelvis_roll_deg = 0.0;
  in.max_joint_velocity_rad_s = 0.01;
  in.ankle_pitch_tracking_abs_max_rad = 0.05;
  in.stable_duration_s = 2.0;
  in.pitch_min_deg = 5.0;
  in.pitch_max_deg = 9.0;
  in.roll_abs_max_deg = 2.0;
  in.max_joint_velocity_limit_rad_s = 0.2;
  in.ankle_pitch_tracking_limit_rad = 0.15;
  in.required_stable_s = 2.0;
  return in;
}

static SupportStepGateInputs ReadySupportStepInputs()
{
  SupportStepGateInputs in;
  in.mode_enabled = true;
  in.policy_active = true;
  in.step_already_started = false;
  in.state_fresh = true;
  in.tilt_trend_ready = true;
  in.policy_elapsed_s = 20.0;
  in.min_policy_elapsed_s = 20.0;
  in.latest_policy_elapsed_s = 40.0;
  in.left_ankle_pitch_tracking_abs_rad = 0.10;
  in.right_ankle_pitch_tracking_abs_rad = 0.10;
  in.waist_pitch_tracking_abs_rad = 0.10;
  in.max_joint_velocity_rad_s = 0.02;
  in.pelvis_tilt_deg = 1.0;
  in.recent_tilt_rise_deg = 0.02;
  in.measurements_stable_s = 5.0;
  return in;
}

void TestEntryAcceptedWhenReady()
{
  const auto d = EvaluateZmqPolicyEntryGate(ReadyInputs());
  EXPECT(d.accepted, d.reason);
  std::cout << "  ok TestEntryAcceptedWhenReady\n";
}

void TestNonZmqAlwaysAccepts()
{
  // On the motion-file path the ZMQ gate is not applicable; the caller's
  // legacy startup gates still run independently.
  ZmqReadinessInputs in;             // all defaults, input_type_is_zmq=false
  in.input_type_is_zmq = false;
  const auto d = EvaluateZmqPolicyEntryGate(in);
  EXPECT(d.accepted, "non-zmq path must not be blocked by the ZMQ gate");
  std::cout << "  ok TestNonZmqAlwaysAccepts\n";
}

void TestRejectNoBodyReference()
{
  auto in = ReadyInputs();
  in.has_body_reference = false;
  const auto d = EvaluateZmqPolicyEntryGate(in);
  EXPECT(!d.accepted, "must refuse policy with no body reference");
  std::cout << "  ok TestRejectNoBodyReference (" << d.reason << ")\n";
}

void TestRejectBeforeWarmup()
{
  auto in = ReadyInputs();
  in.body_frames_received = 39;      // one below threshold
  const auto d = EvaluateZmqPolicyEntryGate(in);
  EXPECT(!d.accepted, "must refuse policy before warmup frame count");
  // Boundary: exactly at threshold is allowed.
  in.body_frames_received = 40;
  EXPECT(EvaluateZmqPolicyEntryGate(in).accepted,
         "warmup boundary (== threshold) must be accepted");
  std::cout << "  ok TestRejectBeforeWarmup (" << d.reason << ")\n";
}

void TestRejectStaleAtEntry()
{
  auto in = ReadyInputs();
  in.reference_age_s = 0.6;          // older than the 0.5 s ceiling
  EXPECT(!EvaluateZmqPolicyEntryGate(in).accepted,
         "must refuse policy when reference is stale at entry");
  // Fail closed on a never-seen reference (negative age).
  in.reference_age_s = -1.0;
  const auto d = EvaluateZmqPolicyEntryGate(in);
  EXPECT(!d.accepted, "negative age (never received) must refuse entry");
  std::cout << "  ok TestRejectStaleAtEntry (" << d.reason << ")\n";
}

void TestRejectUntilContinuouslyFresh()
{
  auto in = ReadyInputs();
  // A single recovered frame (age is fresh) must NOT re-arm policy if the
  // continuity window has not been satisfied yet.
  in.continuous_fresh_s = 0.1;       // below the 0.8 s requirement
  EXPECT(!EvaluateZmqPolicyEntryGate(in).accepted,
         "one fresh frame must not re-arm policy without continuity");
  // A reset continuity window (negative) also refuses.
  in.continuous_fresh_s = -1.0;
  const auto d = EvaluateZmqPolicyEntryGate(in);
  EXPECT(!d.accepted, "reset continuity window must refuse entry");
  std::cout << "  ok TestRejectUntilContinuouslyFresh (" << d.reason << ")\n";
}

void TestStrictVelocityRequiresExplicitFrame()
{
  auto in = ReadyInputs();
  in.strict_velocity       = true;
  in.has_explicit_velocity = false;  // no explicit-velocity frame accepted yet
  const auto d = EvaluateZmqPolicyEntryGate(in);
  EXPECT(!d.accepted, "strict mode must refuse without an explicit-vel frame");
  // Non-strict mode does not require the explicit-velocity frame.
  in.strict_velocity = false;
  EXPECT(EvaluateZmqPolicyEntryGate(in).accepted,
         "non-strict mode must not require an explicit-velocity frame");
  std::cout << "  ok TestStrictVelocityRequiresExplicitFrame (" << d.reason
            << ")\n";
}

void TestEntryCheckOrdering()
{
  // When several conditions fail at once, the reason must name the FIRST
  // (most fundamental) one: no body reference outranks warmup/stale/etc.
  ZmqReadinessInputs in = ReadyInputs();
  in.has_body_reference   = false;
  in.body_frames_received = 0;
  in.reference_age_s      = 5.0;
  const auto d = EvaluateZmqPolicyEntryGate(in);
  EXPECT(!d.accepted, "multiple failures must still refuse");
  EXPECT(d.reason.find("body reference") != std::string::npos,
         "first unmet reason should be the missing body reference");
  std::cout << "  ok TestEntryCheckOrdering (" << d.reason << ")\n";
}

void TestPoseEntryAcceptedAtBoundaries()
{
  auto in = ReadyPoseInputs();
  EXPECT(EvaluateSupportedPoseEntryGate(in).accepted,
         "ready pose entry must be accepted");
  in.pelvis_pitch_deg = 5.0;
  in.pelvis_roll_deg = -2.0;
  EXPECT(EvaluateSupportedPoseEntryGate(in).accepted,
         "lower pitch and roll boundaries must be inclusive");
  in.pelvis_pitch_deg = 9.0;
  in.pelvis_roll_deg = 2.0;
  EXPECT(EvaluateSupportedPoseEntryGate(in).accepted,
         "upper pitch and roll boundaries must be inclusive");
  std::cout << "  ok TestPoseEntryAcceptedAtBoundaries\n";
}

void TestPoseEntryRejectsUnsafeOrUnstableState()
{
  auto in = ReadyPoseInputs();
  in.state_fresh = false;
  EXPECT(!EvaluateSupportedPoseEntryGate(in).accepted,
         "stale robot state must refuse pose entry");

  in = ReadyPoseInputs();
  in.pelvis_pitch_deg = 4.99;
  EXPECT(!EvaluateSupportedPoseEntryGate(in).accepted,
         "pitch below the entry window must be refused");

  in = ReadyPoseInputs();
  in.pelvis_pitch_deg = 9.01;
  EXPECT(!EvaluateSupportedPoseEntryGate(in).accepted,
         "pitch above the entry window must be refused");

  in = ReadyPoseInputs();
  in.pelvis_roll_deg = -2.01;
  EXPECT(!EvaluateSupportedPoseEntryGate(in).accepted,
         "roll outside the entry window must be refused");

  in = ReadyPoseInputs();
  in.max_joint_velocity_rad_s = 0.2;
  EXPECT(!EvaluateSupportedPoseEntryGate(in).accepted,
         "velocity at the ceiling must be refused");

  in = ReadyPoseInputs();
  in.ankle_pitch_tracking_abs_max_rad = 0.151;
  EXPECT(!EvaluateSupportedPoseEntryGate(in).accepted,
         "large pre-policy ankle error must be refused");
  in.ankle_pitch_tracking_abs_max_rad = 0.15;
  EXPECT(EvaluateSupportedPoseEntryGate(in).accepted,
         "ankle tracking threshold must be inclusive");

  in = ReadyPoseInputs();
  in.stable_duration_s = 1.99;
  const auto d = EvaluateSupportedPoseEntryGate(in);
  EXPECT(!d.accepted, "short stability window must be refused");
  EXPECT(d.reason.find("stable") != std::string::npos,
         "stability failure must name the stability gate");
  std::cout << "  ok TestPoseEntryRejectsUnsafeOrUnstableState\n";
}

void TestGroundFaultResetRequiresRecoveredEntry()
{
  GroundFaultResetInputs in;
  in.fault_latched = true;
  in.state_fresh = true;
  in.tilt_deg = 7.0;
  in.max_tilt_deg = 20.0;
  in.pose_entry_ready = true;
  EXPECT(EvaluateGroundFaultResetGate(in).accepted,
         "a latched fault may reset only after measured entry recovery");

  in.fault_latched = false;
  EXPECT(!EvaluateGroundFaultResetGate(in).accepted,
         "reset without a latched fault must be rejected");
  in.fault_latched = true;
  in.state_fresh = false;
  EXPECT(!EvaluateGroundFaultResetGate(in).accepted,
         "stale state must reject reset");
  in.state_fresh = true;
  in.tilt_deg = 20.0;
  EXPECT(!EvaluateGroundFaultResetGate(in).accepted,
         "tilt at the reset ceiling must be rejected");
  in.tilt_deg = 7.0;
  in.pose_entry_ready = false;
  EXPECT(!EvaluateGroundFaultResetGate(in).accepted,
         "an unstable or misaligned pose must reject reset");
  std::cout << "  ok TestGroundFaultResetRequiresRecoveredEntry\n";
}

void TestSupportStepAcceptedOnlyInsideRequestWindow()
{
  auto in = ReadySupportStepInputs();
  EXPECT(EvaluateSupportStepRequestGate(in).accepted,
         "ready support step must be accepted at the earliest boundary");
  in.policy_elapsed_s = 40.0;
  EXPECT(EvaluateSupportStepRequestGate(in).accepted,
         "latest support-step boundary must be inclusive");

  in = ReadySupportStepInputs();
  in.policy_elapsed_s = 19.99;
  EXPECT(!EvaluateSupportStepRequestGate(in).accepted,
         "immutable-support interval must block an early release");
  in.policy_elapsed_s = 40.01;
  EXPECT(!EvaluateSupportStepRequestGate(in).accepted,
         "request after the complete plateau can no longer fit must fail");
  std::cout << "  ok TestSupportStepAcceptedOnlyInsideRequestWindow\n";
}

void TestSupportStepRejectsUnsafeMeasurements()
{
  auto in = ReadySupportStepInputs();
  in.state_fresh = false;
  EXPECT(!EvaluateSupportStepRequestGate(in).accepted,
         "stale state must reject support transfer");

  in = ReadySupportStepInputs();
  in.left_ankle_pitch_tracking_abs_rad = 0.201;
  EXPECT(!EvaluateSupportStepRequestGate(in).accepted,
         "either ankle beyond 0.20 rad must reject support transfer");
  in.left_ankle_pitch_tracking_abs_rad = 0.20;
  EXPECT(EvaluateSupportStepRequestGate(in).accepted,
         "ankle boundary must be inclusive");

  in = ReadySupportStepInputs();
  in.waist_pitch_tracking_abs_rad = 0.201;
  EXPECT(!EvaluateSupportStepRequestGate(in).accepted,
         "waist error beyond 0.20 rad must reject support transfer");
  in.waist_pitch_tracking_limit_rad = -1.0;
  in.waist_pitch_tracking_abs_rad = 0.420;
  EXPECT(EvaluateSupportStepRequestGate(in).accepted,
         "disabled waist gate must retain telemetry without blocking");

  in = ReadySupportStepInputs();
  in.max_joint_velocity_rad_s = 0.051;
  EXPECT(!EvaluateSupportStepRequestGate(in).accepted,
         "joint speed above 0.05 rad/s must reject support transfer");

  in = ReadySupportStepInputs();
  in.pelvis_tilt_deg = 3.01;
  EXPECT(!EvaluateSupportStepRequestGate(in).accepted,
         "tilt above 3 degrees must reject support transfer");

  in = ReadySupportStepInputs();
  in.tilt_trend_ready = false;
  EXPECT(!EvaluateSupportStepRequestGate(in).accepted,
         "a missing trend window must fail closed");
  in.tilt_trend_ready = true;
  in.recent_tilt_rise_deg = 0.11;
  EXPECT(!EvaluateSupportStepRequestGate(in).accepted,
         "increasing tilt must reject support transfer");

  in = ReadySupportStepInputs();
  in.measurements_stable_s = 4.99;
  EXPECT(!EvaluateSupportStepRequestGate(in).accepted,
         "measurements must pass continuously for five seconds");
  std::cout << "  ok TestSupportStepRejectsUnsafeMeasurements\n";
}

void TestSupportStepIsSingleShotAndPlateauIsBounded()
{
  auto in = ReadySupportStepInputs();
  in.mode_enabled = false;
  EXPECT(!EvaluateSupportStepRequestGate(in).accepted,
         "ordinary supported-policy modes must not accept support_step");
  in.mode_enabled = true;
  in.policy_active = false;
  EXPECT(!EvaluateSupportStepRequestGate(in).accepted,
         "support_step outside policy must fail");
  in.policy_active = true;
  in.step_already_started = true;
  EXPECT(!EvaluateSupportStepRequestGate(in).accepted,
         "a second support step must be rejected");

  EXPECT(!SupportStepPlateauComplete(false, 100.0, 20.0),
         "an unmarked run must not complete a support-step plateau");
  EXPECT(!SupportStepPlateauComplete(true, 19.99, 20.0),
         "plateau must run for the complete configured duration");
  EXPECT(SupportStepPlateauComplete(true, 20.0, 20.0),
         "plateau completion boundary must be inclusive");
  std::cout << "  ok TestSupportStepIsSingleShotAndPlateauIsBounded\n";
}

void TestM2LoadedEquilibriumFailsSupportStepGate()
{
  auto in = ReadySupportStepInputs();
  in.waist_pitch_tracking_limit_rad = -1.0;
  in.left_ankle_pitch_tracking_abs_rad = 0.494;
  in.right_ankle_pitch_tracking_abs_rad = 0.120;
  in.waist_pitch_tracking_abs_rad = 0.420;
  const auto d = EvaluateSupportStepRequestGate(in);
  EXPECT(!d.accepted,
         "the M2 taut-sling equilibrium must not authorize sling release");
  EXPECT(d.reason.find("ankle-pitch") != std::string::npos,
         "M2 rejection must identify the first measured blocker");
  std::cout << "  ok TestM2LoadedEquilibriumFailsSupportStepGate\n";
}

void TestGroundLoadDebugRequiresLiveMeasuredState()
{
  EXPECT(ShouldPublishGroundLoadDebug(
             /*snapshot_available=*/true,
             /*all_state_fresh=*/true,
             /*debug_publisher_enabled=*/true),
         "live measured state and an enabled publisher must emit x2_debug");
  EXPECT(!ShouldPublishGroundLoadDebug(false, true, true),
         "missing RobotState snapshot must suppress x2_debug");
  EXPECT(!ShouldPublishGroundLoadDebug(true, false, true),
         "stale HAL state must not be republished as fresh x2_debug");
  EXPECT(!ShouldPublishGroundLoadDebug(true, true, false),
         "disabled debug output must remain a no-op");
  std::cout << "  ok TestGroundLoadDebugRequiresLiveMeasuredState\n";
}

void TestSupportedPolicyStaleTrips()
{
  const double stale_s = 0.5;
  // Fresh reference -> not stale -> keep ticking policy.
  EXPECT(!ZmqReferenceStale(/*watchdog_active=*/true, /*age=*/0.1, stale_s),
         "fresh reference must not trip the supported-policy return");
  // Aged past threshold -> stale -> bounded return.
  EXPECT(ZmqReferenceStale(true, 0.5, stale_s),
         "age == threshold must trip (>=)");
  EXPECT(ZmqReferenceStale(true, 0.9, stale_s),
         "age past threshold must trip the supported-policy return");
  std::cout << "  ok TestSupportedPolicyStaleTrips\n";
}

void TestStaleFailsClosedOnNeverReceived()
{
  // A negative age (source never delivered a frame) must be treated as stale:
  // fail closed rather than tick powered policy on a phantom reference.
  EXPECT(ZmqReferenceStale(/*watchdog_active=*/true, /*age=*/-1.0, 0.5),
         "never-received reference must fail closed (treated as stale)");
  std::cout << "  ok TestStaleFailsClosedOnNeverReceived\n";
}

void TestStaleNoopWhenWatchdogInactive()
{
  // On the non-zmq path (or --disable-pose-ref-watchdog, or stale_s<=0) the
  // watchdog is inactive and must NEVER trip -- even on a negative age.
  EXPECT(!ZmqReferenceStale(/*watchdog_active=*/false, /*age=*/-1.0, 0.5),
         "inactive watchdog must not trip on negative age");
  EXPECT(!ZmqReferenceStale(false, 100.0, 0.5),
         "inactive watchdog must not trip on a huge age");
  std::cout << "  ok TestStaleNoopWhenWatchdogInactive\n";
}

int main()
{
  std::cout << "agi_x2 supported-policy live-reference gate tests\n";
  TestEntryAcceptedWhenReady();
  TestNonZmqAlwaysAccepts();
  TestRejectNoBodyReference();
  TestRejectBeforeWarmup();
  TestRejectStaleAtEntry();
  TestRejectUntilContinuouslyFresh();
  TestStrictVelocityRequiresExplicitFrame();
  TestEntryCheckOrdering();
  TestPoseEntryAcceptedAtBoundaries();
  TestPoseEntryRejectsUnsafeOrUnstableState();
  TestGroundFaultResetRequiresRecoveredEntry();
  TestSupportStepAcceptedOnlyInsideRequestWindow();
  TestSupportStepRejectsUnsafeMeasurements();
  TestSupportStepIsSingleShotAndPlateauIsBounded();
  TestM2LoadedEquilibriumFailsSupportStepGate();
  TestGroundLoadDebugRequiresLiveMeasuredState();
  TestSupportedPolicyStaleTrips();
  TestStaleFailsClosedOnNeverReceived();
  TestStaleNoopWhenWatchdogInactive();
  std::cout << "all OK\n";
  return 0;
}
