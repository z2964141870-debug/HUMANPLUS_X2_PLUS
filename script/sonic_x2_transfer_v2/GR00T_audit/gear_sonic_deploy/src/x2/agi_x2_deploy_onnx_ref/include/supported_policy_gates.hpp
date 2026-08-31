/**
 * @file supported_policy_gates.hpp
 * @brief Pure (no-ROS, no-ONNX) decision logic for supported-policy entry and
 *        stale-reference gates.
 *
 * The X2 deploy state machine (``x2_deploy_onnx_ref.cpp``) is a ROS 2 node
 * that owns ONNX inference, HAL IO, and hardware timers -- none of which can
 * be instantiated in a workstation unit test. To keep the powered-safety
 * decisions testable WITHOUT a robot, the decisions that gate a powered
 * transition are factored out here as plain functions over plain-old-data:
 *
 *   1. ``EvaluateZmqPolicyEntryGate`` -- may the operator's ``policy`` request
 *      be accepted, given the live ZMQ reference readiness signals? (called
 *      from ``RequestSupportedPolicy``.)
 *   2. ``EvaluateSupportedPoseEntryGate`` -- is the reconstructed pelvis in a
 *      measured, continuously-stable entry window?
 *   3. ``EvaluateSupportStepRequestGate`` -- may one manually executed,
 *      timestamped support-transfer step begin during a bounded policy probe?
 *   4. ``ShouldPublishGroundLoadDebug`` -- may ``GROUND_LOAD_HOLD`` publish
 *      measured robot telemetry without making stale state look fresh?
 *   5. ``ZmqReferenceStale`` -- during ``SUPPORTED_POLICY``, has the live pose
 *      reference starved and therefore must trigger the bounded return to the
 *      captured static hold? (called from the ``SUPPORTED_POLICY`` case.)
 *
 * Both are ``inline`` and header-only so the deploy binary and the offline
 * test target compile the exact same logic. See test/test_supported_policy.cpp.
 */

#ifndef AGI_X2_SUPPORTED_POLICY_GATES_HPP
#define AGI_X2_SUPPORTED_POLICY_GATES_HPP

#include <cmath>
#include <cstdint>
#include <string>

namespace agi_x2 {

/// Outcome of a policy-entry gate evaluation.
struct PolicyEntryDecision {
  bool        accepted{false};
  std::string reason;
};

/// Snapshot of the ZMQ reference readiness at the instant the operator
/// requests powered policy entry. Populated by the deploy from
/// ``ZmqPoseInputSource`` accessors + CLI thresholds; populated directly by
/// the unit tests. No ROS, no ONNX, no hardware -- just numbers.
struct ZmqReadinessInputs {
  /// True when ``--input-type=zmq``. On the motion_file path the ZMQ gate is
  /// not applicable and always accepts (the legacy startup gates still run in
  /// the caller).
  bool    input_type_is_zmq{false};
  /// ``ZmqPoseInputSource::has_body_reference()``.
  bool    has_body_reference{false};
  /// ``ZmqPoseInputSource::body_frames_received()`` -- ACCEPTED body frames
  /// only (token/hand-only and rejected frames excluded by construction).
  int64_t body_frames_received{0};
  /// Minimum accepted body frames before entry (``--zmq-warmup-body-frames``).
  int64_t warmup_body_frames{40};
  /// ``now - LastReceivedMonotonicS()``. Negative means no frame ever arrived.
  double  reference_age_s{-1.0};
  /// Entry freshness ceiling (``--zmq-entry-max-age-s``, default 0.5 s).
  double  entry_max_age_s{0.5};
  /// How long the reference has been CONTINUOUSLY fresh (age below the stale
  /// threshold on every tick). Negative means the continuity window is not
  /// currently satisfied (a gap reset it). This is what prevents a single
  /// freshly-recovered frame from immediately re-arming policy entry.
  double  continuous_fresh_s{-1.0};
  /// Required continuous-fresh window (seconds). Default 0.8 s ~= 40 frames
  /// at 50 Hz.
  double  min_continuous_fresh_s{0.8};
  /// ``--strict-reference-velocity`` active on this run.
  bool    strict_velocity{false};
  /// ``ZmqPoseInputSource::has_explicit_reference_velocity()``.
  bool    has_explicit_velocity{false};
};

/// Decide whether a powered policy-entry request is allowed on the ZMQ path.
///
/// The checks are ordered from cheapest/most-fundamental to most-specific so
/// the returned reason names the FIRST unmet condition. On the non-ZMQ path
/// this always accepts -- the caller's legacy startup gates (state machine,
/// HAL freshness, post-handoff stability) still apply independently.
inline PolicyEntryDecision EvaluateZmqPolicyEntryGate(
    const ZmqReadinessInputs& in)
{
  if (!in.input_type_is_zmq) {
    return {true, "non-zmq input: ZMQ readiness gate not applicable"};
  }
  if (!in.has_body_reference) {
    return {false, "no body reference received yet"};
  }
  if (in.body_frames_received < in.warmup_body_frames) {
    return {false, "warmup incomplete: body_frames below threshold"};
  }
  if (in.reference_age_s < 0.0 || in.reference_age_s > in.entry_max_age_s) {
    return {false, "reference stale: age exceeds entry ceiling"};
  }
  if (in.continuous_fresh_s < in.min_continuous_fresh_s) {
    return {false, "reference not continuously fresh long enough"};
  }
  if (in.strict_velocity && !in.has_explicit_velocity) {
    return {false,
            "strict velocity mode: no explicit-velocity frame accepted yet"};
  }
  return {true, "ready"};
}

/// Reconstructed-pelvis and motion snapshot used to gate a supported policy
/// pulse. The caller maintains ``stable_duration_s`` by resetting its timer
/// whenever any instantaneous condition leaves the configured window.
struct SupportedPoseEntryInputs {
  bool   state_fresh{false};
  double pelvis_pitch_deg{0.0};
  double pelvis_roll_deg{0.0};
  double max_joint_velocity_rad_s{0.0};
  double ankle_pitch_tracking_abs_max_rad{0.0};
  double stable_duration_s{0.0};
  double pitch_min_deg{-180.0};
  double pitch_max_deg{180.0};
  double roll_abs_max_deg{180.0};
  double max_joint_velocity_limit_rad_s{0.2};
  /// Non-positive disables this check for backward-compatible launchers.
  double ankle_pitch_tracking_limit_rad{-1.0};
  double required_stable_s{0.0};
};

/// Require software-measured pelvis orientation and joint motion to remain in
/// the configured window before entering powered policy control. Boundaries
/// are inclusive for orientation and stable duration; the velocity ceiling is
/// exclusive, matching the existing ``max_vel >= 0.2`` rejection behavior.
inline PolicyEntryDecision EvaluateSupportedPoseEntryGate(
    const SupportedPoseEntryInputs& in)
{
  if (!in.state_fresh) {
    return {false, "robot state is stale"};
  }
  if (in.pelvis_pitch_deg < in.pitch_min_deg
      || in.pelvis_pitch_deg > in.pitch_max_deg) {
    return {false, "pelvis pitch outside entry window"};
  }
  if (std::abs(in.pelvis_roll_deg) > in.roll_abs_max_deg) {
    return {false, "pelvis roll outside entry window"};
  }
  if (in.max_joint_velocity_rad_s >= in.max_joint_velocity_limit_rad_s) {
    return {false, "joint velocity outside entry window"};
  }
  if (in.ankle_pitch_tracking_limit_rad > 0.0
      && in.ankle_pitch_tracking_abs_max_rad
             > in.ankle_pitch_tracking_limit_rad) {
    return {false, "ankle-pitch tracking error outside entry window"};
  }
  if (in.stable_duration_s < in.required_stable_s) {
    return {false, "pelvis entry window not stable long enough"};
  }
  return {true, "ready"};
}

/// Measurements and timing used to admit exactly one manually executed
/// support-transfer step during a bounded fixed-reference policy probe. The
/// helper only decides whether the event marker may be accepted; it never
/// changes a command, gain, target, or control owner.
struct SupportStepGateInputs {
  bool   mode_enabled{false};
  bool   policy_active{false};
  bool   step_already_started{false};
  bool   state_fresh{false};
  bool   tilt_trend_ready{false};
  double policy_elapsed_s{0.0};
  double min_policy_elapsed_s{20.0};
  double latest_policy_elapsed_s{40.0};
  double left_ankle_pitch_tracking_abs_rad{0.0};
  double right_ankle_pitch_tracking_abs_rad{0.0};
  double waist_pitch_tracking_abs_rad{0.0};
  double max_joint_velocity_rad_s{0.0};
  double pelvis_tilt_deg{0.0};
  /// Current tilt minus the minimum tilt in the recent trend window.
  double recent_tilt_rise_deg{0.0};
  double ankle_pitch_tracking_limit_rad{0.20};
  // A negative limit records waist tracking without using it as a gate. The
  // upper-body sling applies an unavoidable waist moment on X2, so release
  // readiness must not require that load-dependent offset to approach zero.
  double waist_pitch_tracking_limit_rad{0.20};
  double max_joint_velocity_limit_rad_s{0.05};
  double max_pelvis_tilt_deg{3.0};
  double max_recent_tilt_rise_deg{0.10};
  double measurements_stable_s{0.0};
  double required_stable_s{5.0};
};

/// Instantaneous measurement portion of the support-step gate. The caller
/// uses this result to maintain the continuous-stability timer.
inline PolicyEntryDecision EvaluateSupportStepMeasurements(
    const SupportStepGateInputs& in)
{
  if (!in.state_fresh) {
    return {false, "robot state is stale"};
  }
  if (in.left_ankle_pitch_tracking_abs_rad
          > in.ankle_pitch_tracking_limit_rad
      || in.right_ankle_pitch_tracking_abs_rad
             > in.ankle_pitch_tracking_limit_rad) {
    return {false, "ankle-pitch tracking error exceeds support-step gate"};
  }
  if (in.waist_pitch_tracking_limit_rad >= 0.0
      && in.waist_pitch_tracking_abs_rad
             > in.waist_pitch_tracking_limit_rad) {
    return {false, "waist-pitch tracking error exceeds support-step gate"};
  }
  if (in.max_joint_velocity_rad_s
      > in.max_joint_velocity_limit_rad_s) {
    return {false, "joint velocity exceeds support-step gate"};
  }
  if (in.pelvis_tilt_deg > in.max_pelvis_tilt_deg) {
    return {false, "pelvis tilt exceeds support-step gate"};
  }
  if (!in.tilt_trend_ready) {
    return {false, "pelvis tilt trend window is not ready"};
  }
  if (in.recent_tilt_rise_deg > in.max_recent_tilt_rise_deg) {
    return {false, "pelvis tilt is increasing"};
  }
  return {true, "ready"};
}

/// Full operator-request gate. Boundaries are inclusive: a request at the
/// earliest/latest allowed time and measurements exactly at a ceiling pass.
inline PolicyEntryDecision EvaluateSupportStepRequestGate(
    const SupportStepGateInputs& in)
{
  if (!in.mode_enabled) {
    return {false, "support-step mode is disabled"};
  }
  if (!in.policy_active) {
    return {false, "supported policy is not active"};
  }
  if (in.step_already_started) {
    return {false, "support step already started"};
  }
  if (in.policy_elapsed_s < in.min_policy_elapsed_s) {
    return {false, "immutable-support interval is not complete"};
  }
  if (in.policy_elapsed_s > in.latest_policy_elapsed_s) {
    return {false, "support-step request window has closed"};
  }
  const auto measurements = EvaluateSupportStepMeasurements(in);
  if (!measurements.accepted) {
    return measurements;
  }
  if (in.measurements_stable_s < in.required_stable_s) {
    return {false, "support-step measurements are not stable long enough"};
  }
  return {true, "ready"};
}

inline bool SupportStepPlateauComplete(bool step_started,
                                       double step_elapsed_s,
                                       double plateau_seconds)
{
  return step_started && plateau_seconds > 0.0
      && step_elapsed_s >= plateau_seconds;
}

/// Inputs for an operator-requested reset of a latched supported static-PD
/// fault. Resetting never changes the command or control owner; it only allows
/// a later policy request after the robot has returned to the same measured
/// entry envelope used for normal policy entry.
struct GroundFaultResetInputs {
  bool   fault_latched{false};
  bool   state_fresh{false};
  double tilt_deg{-1.0};
  double max_tilt_deg{20.0};
  bool   pose_entry_ready{false};
};

inline PolicyEntryDecision EvaluateGroundFaultResetGate(
    const GroundFaultResetInputs& in)
{
  if (!in.fault_latched) {
    return {false, "no ground-load fault is latched"};
  }
  if (!in.state_fresh) {
    return {false, "robot state is stale"};
  }
  if (in.tilt_deg < 0.0 || in.tilt_deg >= in.max_tilt_deg) {
    return {false, "pelvis tilt outside reset envelope"};
  }
  if (!in.pose_entry_ready) {
    return {false, "supported pose entry gate is not ready"};
  }
  return {true, "ready"};
}

/// Publish pre-policy robot telemetry only while the measured state is live.
/// The proxy measures freshness from x2_debug arrival time, so continuing to
/// emit cached RobotState after HAL feedback stops would incorrectly make a
/// stale robot look current. The publisher itself remains optional.
inline bool ShouldPublishGroundLoadDebug(bool snapshot_available,
                                         bool all_state_fresh,
                                         bool debug_publisher_enabled)
{
  return snapshot_available && all_state_fresh && debug_publisher_enabled;
}

/// Decide whether the live pose reference has starved during
/// ``SUPPORTED_POLICY`` and must therefore trigger the bounded return to the
/// captured static hold.
///
/// ``watchdog_active`` mirrors ``pose_ref_watchdog_active_`` in the deploy
/// (true only on the ZMQ path with a positive stale threshold and the
/// watchdog not disabled). A negative ``reference_age_s`` (no frame ever, or
/// the source reports the "never received" sentinel) is treated as stale --
/// fail closed.
inline bool ZmqReferenceStale(bool   watchdog_active,
                              double reference_age_s,
                              double pose_ref_stale_s)
{
  if (!watchdog_active) return false;
  if (reference_age_s < 0.0) return true;
  return reference_age_s >= pose_ref_stale_s;
}

}  // namespace agi_x2

#endif  // AGI_X2_SUPPORTED_POLICY_GATES_HPP
