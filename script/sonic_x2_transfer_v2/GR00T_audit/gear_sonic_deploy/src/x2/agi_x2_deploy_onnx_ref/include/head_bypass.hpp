/**
 * @file head_bypass.hpp
 * @brief Optional reference override for the two X2 head joints.
 */

#ifndef AGI_X2_HEAD_BYPASS_HPP
#define AGI_X2_HEAD_BYPASS_HPP

#include "policy_parameters.hpp"
#include "reference_motion.hpp"

#include <array>
#include <cmath>

namespace agi_x2 {

inline constexpr std::array<int, 2> kBypassedHeadMjDofs = {29, 30};

inline double ApplyHeadBypass(std::array<double, NUM_DOFS>& target_pos_mj,
                              const ReferenceFrame&         ref)
{
  double max_delta = 0.0;
  for (const int mj : kBypassedHeadMjDofs) {
    const double delta = std::fabs(target_pos_mj[mj] - ref.joint_pos_mj[mj]);
    if (delta > max_delta) max_delta = delta;
    target_pos_mj[mj] = ref.joint_pos_mj[mj];
  }
  return max_delta;
}

}  // namespace agi_x2

#endif  // AGI_X2_HEAD_BYPASS_HPP
