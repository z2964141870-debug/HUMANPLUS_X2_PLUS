/**
 * @file startup_ramp.hpp
 * @brief Pure helpers for the suspended Sonic cold-start trajectory.
 */

#ifndef AGI_X2_STARTUP_RAMP_HPP
#define AGI_X2_STARTUP_RAMP_HPP

#include "policy_parameters.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace agi_x2 {

inline double StartupSmoothStep(double elapsed_s, double duration_s)
{
  if (duration_s <= 0.0) return 1.0;
  const double u = std::clamp(elapsed_s / duration_s, 0.0, 1.0);
  return u * u * (3.0 - 2.0 * u);
}

inline double StartupPoseRampDuration(
    const std::array<double, NUM_DOFS>& anchor,
    const std::array<double, NUM_DOFS>& desired,
    double minimum_seconds,
    double maximum_rate_rad_s)
{
  double max_delta = 0.0;
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    max_delta = std::max(max_delta, std::abs(desired[i] - anchor[i]));
  }
  return std::max(minimum_seconds, max_delta / maximum_rate_rad_s);
}

inline double StartupPoseRampDuration(
    const std::array<double, NUM_DOFS>& anchor,
    double minimum_seconds,
    double maximum_rate_rad_s)
{
  return StartupPoseRampDuration(
      anchor, default_angles, minimum_seconds, maximum_rate_rad_s);
}

inline std::array<double, NUM_DOFS> StartupPoseTarget(
    const std::array<double, NUM_DOFS>& anchor,
    const std::array<double, NUM_DOFS>& desired,
    double alpha)
{
  std::array<double, NUM_DOFS> target{};
  const double a = std::clamp(alpha, 0.0, 1.0);
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    target[i] = anchor[i] + a * (desired[i] - anchor[i]);
  }
  return target;
}

inline std::array<double, NUM_DOFS> StartupPoseTarget(
    const std::array<double, NUM_DOFS>& anchor,
    double alpha)
{
  return StartupPoseTarget(anchor, default_angles, alpha);
}

inline double SupportedPolicyJointLimit(
    std::size_t joint_index,
    double leg_limit,
    double waist_limit,
    double arm_limit,
    double head_limit)
{
  // X2 MuJoCo order: legs 0..11, waist 12..14, arms 15..28, head 29..30.
  if (joint_index < 12) return leg_limit;
  if (joint_index < 15) return waist_limit;
  if (joint_index < 29) return arm_limit;
  return head_limit;
}

inline std::array<double, NUM_DOFS> SupportedPolicyTarget(
    const std::array<double, NUM_DOFS>& static_target,
    const std::array<double, NUM_DOFS>& policy_target,
    double alpha,
    double leg_limit = 0.02,
    double waist_limit = 0.02,
    double arm_limit = 0.03,
    double head_limit = 0.03)
{
  std::array<double, NUM_DOFS> target{};
  const double a = std::clamp(alpha, 0.0, 1.0);
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    const double limit = SupportedPolicyJointLimit(
        i, leg_limit, waist_limit, arm_limit, head_limit);
    const double delta = std::clamp(
        policy_target[i] - static_target[i], -limit, limit);
    target[i] = static_target[i] + a * delta;
  }
  return target;
}

inline std::array<double, NUM_DOFS> SupportedPolicyTargetAroundDefault(
    const std::array<double, NUM_DOFS>& static_target,
    const std::array<double, NUM_DOFS>& policy_target,
    double alpha,
    double leg_limit,
    double waist_limit,
    double arm_limit,
    double head_limit)
{
  std::array<double, NUM_DOFS> target{};
  const double a = std::clamp(alpha, 0.0, 1.0);
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    const double limit = SupportedPolicyJointLimit(
        i, leg_limit, waist_limit, arm_limit, head_limit);
    const double bounded_policy = default_angles[i] + std::clamp(
        policy_target[i] - default_angles[i], -limit, limit);
    target[i] = static_target[i] + a * (bounded_policy - static_target[i]);
  }
  return target;
}

inline std::array<double, NUM_DOFS> SupportedPolicyReturnTarget(
    const std::array<double, NUM_DOFS>& return_anchor,
    const std::array<double, NUM_DOFS>& static_target,
    double alpha)
{
  std::array<double, NUM_DOFS> target{};
  const double a = StartupSmoothStep(alpha, 1.0);
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    target[i] = return_anchor[i] + a * (static_target[i] - return_anchor[i]);
  }
  return target;
}

inline std::array<double, NUM_DOFS> SlewLimitTarget(
    const std::array<double, NUM_DOFS>& previous,
    const std::array<double, NUM_DOFS>& desired,
    double maximum_rate_rad_s,
    double elapsed_s)
{
  if (maximum_rate_rad_s <= 0.0 || elapsed_s <= 0.0) return desired;
  std::array<double, NUM_DOFS> target{};
  const double maximum_step = maximum_rate_rad_s * elapsed_s;
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    target[i] = previous[i] + std::clamp(
        desired[i] - previous[i], -maximum_step, maximum_step);
  }
  return target;
}

inline std::array<double, NUM_DOFS> AppliedActionFromTarget(
    const std::array<double, NUM_DOFS>& target_mj)
{
  std::array<double, NUM_DOFS> action_il{};
  for (std::size_t mj = 0; mj < NUM_DOFS; ++mj) {
    const std::size_t il = static_cast<std::size_t>(mujoco_to_isaaclab[mj]);
    action_il[il] =
        (target_mj[mj] - default_angles[mj]) / x2_action_scale[mj];
  }
  return action_il;
}

}  // namespace agi_x2

#endif  // AGI_X2_STARTUP_RAMP_HPP
