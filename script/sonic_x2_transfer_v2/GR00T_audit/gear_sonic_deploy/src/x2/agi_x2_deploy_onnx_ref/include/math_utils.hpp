/**
 * @file math_utils.hpp
 * @brief Minimal quaternion / vector helpers used by the X2 obs builder.
 *
 * We only need the operations the Python obs builder uses:
 *   - quat_rotate_inverse(q_wxyz, v)        -> body-frame projection
 *   - quat_relative(q_a_xyzw, q_b_xyzw)     -> q_a^-1 * q_b
 *   - rot6d_from_quat(q_xyzw)               -> first two rows of the matrix
 *
 * All quaternion conventions are explicitly tagged in the function names
 * (`_wxyz` vs `_xyzw`) because the X2 PKLs are scipy-style xyzw while
 * MuJoCo / IMU messages are wxyz, and silently mixing them is the most
 * likely source of "policy looks fine but robot tilts" bugs.
 */

#ifndef AGI_X2_MATH_UTILS_HPP
#define AGI_X2_MATH_UTILS_HPP

#include <array>
#include <cmath>

namespace agi_x2 {

/// Rotate vector v by INVERSE of quaternion q (wxyz convention).
/// Mirrors gear_sonic/scripts/eval_x2_mujoco.py::quat_rotate_inverse, which
/// in turn matches IsaacLab's quat_apply_inverse: v - w*t + cross(u, t)
/// with t = 2 * cross(u, v) and u = (qx, qy, qz).
inline std::array<double, 3> quat_rotate_inverse_wxyz(
    const std::array<double, 4>& q_wxyz,
    const std::array<double, 3>& v)
{
  const double w = q_wxyz[0];
  const double ux = q_wxyz[1], uy = q_wxyz[2], uz = q_wxyz[3];

  // t = 2 * cross(u, v)
  const double tx = 2.0 * (uy * v[2] - uz * v[1]);
  const double ty = 2.0 * (uz * v[0] - ux * v[2]);
  const double tz = 2.0 * (ux * v[1] - uy * v[0]);

  // out = v - w*t + cross(u, t)
  return {
    v[0] - w * tx + (uy * tz - uz * ty),
    v[1] - w * ty + (uz * tx - ux * tz),
    v[2] - w * tz + (ux * ty - uy * tx),
  };
}

/// Rotate vector v by quaternion q (wxyz convention).
inline std::array<double, 3> quat_rotate_wxyz(
    const std::array<double, 4>& q_wxyz,
    const std::array<double, 3>& v)
{
  const std::array<double, 4> q_inv = {
      q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]};
  return quat_rotate_inverse_wxyz(q_inv, v);
}

/// Quaternion multiply (wxyz): out = a * b.
inline std::array<double, 4> quat_mul_wxyz(
    const std::array<double, 4>& a,
    const std::array<double, 4>& b)
{
  return {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  };
}

inline std::array<double, 4> quat_normalize_wxyz(
    const std::array<double, 4>& q)
{
  const double norm = std::sqrt(
      q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (norm <= 1e-12) return {1.0, 0.0, 0.0, 0.0};
  return {q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm};
}

/// Rotation from pelvis to torso for the X2 waist chain:
/// Rz(waist_yaw) * Ry(waist_pitch) * Rx(waist_roll).
inline std::array<double, 4> x2_pelvis_to_torso_quat_wxyz(
    double yaw, double pitch, double roll)
{
  const double hy = 0.5 * yaw;
  const double hp = 0.5 * pitch;
  const double hr = 0.5 * roll;
  const std::array<double, 4> qz = {
      std::cos(hy), 0.0, 0.0, std::sin(hy)};
  const std::array<double, 4> qy = {
      std::cos(hp), 0.0, std::sin(hp), 0.0};
  const std::array<double, 4> qx = {
      std::cos(hr), std::sin(hr), 0.0, 0.0};
  return quat_normalize_wxyz(quat_mul_wxyz(quat_mul_wxyz(qz, qy), qx));
}

struct X2PelvisKinematics {
  std::array<double, 4> quat_wxyz{1.0, 0.0, 0.0, 0.0};
  std::array<double, 3> ang_vel{0.0, 0.0, 0.0};
};

/// Recover pelvis/root state from the torso-mounted IMU and the three waist
/// joints. The Sonic policy was trained with the pelvis IMU in x2_ultra.xml;
/// feeding the hardware torso IMU directly makes waist flexion look like root
/// tilt and causes a false ankle-balancing response.
inline X2PelvisKinematics x2_reconstruct_pelvis_from_torso(
    const std::array<double, 4>& torso_quat_wxyz,
    const std::array<double, 3>& torso_ang_vel,
    double waist_yaw, double waist_pitch, double waist_roll,
    double waist_yaw_vel, double waist_pitch_vel, double waist_roll_vel)
{
  const auto q_pt = x2_pelvis_to_torso_quat_wxyz(
      waist_yaw, waist_pitch, waist_roll);
  const std::array<double, 4> q_tp = {
      q_pt[0], -q_pt[1], -q_pt[2], -q_pt[3]};

  X2PelvisKinematics out;
  out.quat_wxyz = quat_normalize_wxyz(
      quat_mul_wxyz(quat_normalize_wxyz(torso_quat_wxyz), q_tp));

  // Relative waist angular velocity expressed in the final torso frame.
  const auto yaw_t = quat_rotate_inverse_wxyz(
      q_pt, {0.0, 0.0, waist_yaw_vel});
  const auto q_pitch_roll = quat_mul_wxyz(
      x2_pelvis_to_torso_quat_wxyz(0.0, waist_pitch, 0.0),
      x2_pelvis_to_torso_quat_wxyz(0.0, 0.0, waist_roll));
  const auto pitch_t = quat_rotate_inverse_wxyz(
      q_pitch_roll, {0.0, waist_pitch_vel, 0.0});
  const std::array<double, 3> roll_t = {waist_roll_vel, 0.0, 0.0};
  const std::array<double, 3> pelvis_in_torso = {
      torso_ang_vel[0] - yaw_t[0] - pitch_t[0] - roll_t[0],
      torso_ang_vel[1] - yaw_t[1] - pitch_t[1] - roll_t[1],
      torso_ang_vel[2] - yaw_t[2] - pitch_t[2] - roll_t[2],
  };
  out.ang_vel = quat_rotate_wxyz(q_pt, pelvis_in_torso);
  return out;
}

/// Project gravity direction [0,0,-1] world into the body frame.
/// This is what ProprioceptionBuffer.gravity_hist stores.
inline std::array<double, 3> body_frame_gravity_from_quat_wxyz(
    const std::array<double, 4>& q_wxyz)
{
  return quat_rotate_inverse_wxyz(q_wxyz, {0.0, 0.0, -1.0});
}

/// Quaternion multiply (xyzw): out = a * b. Used to compute relative rotations.
inline std::array<double, 4> quat_mul_xyzw(
    const std::array<double, 4>& a,
    const std::array<double, 4>& b)
{
  const double ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const double bx = b[0], by = b[1], bz = b[2], bw = b[3];
  return {
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  };
}

/// Conjugate of a unit quaternion (xyzw): just negate the vector part.
inline std::array<double, 4> quat_conj_xyzw(const std::array<double, 4>& q)
{
  return { -q[0], -q[1], -q[2], q[3] };
}

/// Convert wxyz quaternion to xyzw (scipy / PKL convention).
inline std::array<double, 4> wxyz_to_xyzw(const std::array<double, 4>& q_wxyz)
{
  return { q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0] };
}

/// Convert xyzw quaternion to wxyz (MuJoCo / IMU convention).
inline std::array<double, 4> xyzw_to_wxyz(const std::array<double, 4>& q_xyzw)
{
  return { q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2] };
}

/// Build a 3x3 rotation matrix from a unit quaternion (xyzw, scipy ordering).
/// Returns row-major: out[row*3 + col]. Matches scipy.spatial.transform.Rotation.
inline std::array<double, 9> rotation_matrix_xyzw(
    const std::array<double, 4>& q_xyzw)
{
  const double x = q_xyzw[0], y = q_xyzw[1], z = q_xyzw[2], w = q_xyzw[3];
  const double xx = x * x, yy = y * y, zz = z * z;
  const double xy = x * y, xz = x * z, yz = y * z;
  const double wx = w * x, wy = w * y, wz = w * z;
  return {
    1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
    2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
    2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy),
  };
}

/// 6D rotation representation: first two COLUMNS of the rotation matrix,
/// flattened row-major. Matches IsaacLab's ``motion_anchor_ori_b_mf`` term
/// (gear_sonic/envs/manager_env/mdp/observations.py::motion_anchor_ori_b_mf
/// -> commands.py::root_rot_dif_l_multi_future), which is:
///
///     mat = matrix_from_quat(root_rot_dif).reshape(B, F, 3, 3)
///     rot6 = mat[..., :2].reshape(B, -1)               # first 2 COLUMNS
///
/// Because PyTorch ``mat[..., :2]`` slices the LAST dim of (3, 3), the
/// resulting (3, 2) sub-tensor is the first two COLUMNS of the rotmat
/// (row-index varies fastest under the implicit C-order flatten that
/// follows). Flattened row-major it is:
///
///     [m[0,0], m[0,1], m[1,0], m[1,1], m[2,0], m[2,1]]
///
/// rotation_matrix_xyzw above stores out[row*3 + col] (row-major), so:
///
///     m[0,0] = m[0],  m[0,1] = m[1]
///     m[1,0] = m[3],  m[1,1] = m[4]
///     m[2,0] = m[6],  m[2,1] = m[7]
///
/// Bug history: the previous implementation returned the first two ROWS
/// (m[0..5] = m[0,0], m[0,1], m[0,2], m[1,0], m[1,1], m[1,2]). For the
/// identity rotation that happens to equal ``[1, 0, 0, 0, 1, 0]`` which is
/// also what columns-flat returns for identity, so the unit test in
/// test_obs_builder.cpp passed. Off-axis the two reps disagree: e.g. for a
/// 90-degree yaw the rows-flat rep is [0,-1,0, 1,0,0] while the columns-
/// flat rep is [0,1,0, -1,0,0]. The trained policy expected columns-flat,
/// so the rows-flat C++ output left ``motion_anchor_ori_b_mf`` 6D garbage
/// whenever the reference motion's frame deviated from the robot's by more
/// than a few degrees. This was caught by the slot diff in
/// gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py.
inline std::array<double, 6> rot6d_from_quat_xyzw(
    const std::array<double, 4>& q_xyzw)
{
  const auto m = rotation_matrix_xyzw(q_xyzw);
  return { m[0], m[1], m[3], m[4], m[6], m[7] };
}

/// Extract the world-frame yaw (heading) angle from a unit quaternion (xyzw).
///
/// We compute the world-frame forward vector that the robot's body-X axis
/// gets rotated into by the quaternion (= first column of the rotmat), then
/// take atan2(forward.y, forward.x). This is more robust to non-zero
/// pitch/roll than the naive 2*atan2(qz, qw) shortcut: for a robot tilted
/// 10 deg in pitch the shortcut starts losing yaw resolution; this version
/// stays well-behaved as long as the body-X axis isn't pointing nearly
/// vertical (i.e. the robot isn't lying on its face).
inline double yaw_from_quat_xyzw(const std::array<double, 4>& q_xyzw)
{
  const double x = q_xyzw[0], y = q_xyzw[1], z = q_xyzw[2], w = q_xyzw[3];
  const double fx = 1.0 - 2.0 * (y * y + z * z);
  const double fy = 2.0 * (x * y + w * z);
  return std::atan2(fy, fx);
}

/// Extract the world-frame yaw from a unit quaternion (wxyz, IMU convention).
inline double yaw_from_quat_wxyz(const std::array<double, 4>& q_wxyz)
{
  return yaw_from_quat_xyzw(wxyz_to_xyzw(q_wxyz));
}

/// Build a pure-yaw rotation quaternion (xyzw) Rz(yaw_rad). For yaw=0 this
/// returns the identity quaternion (0,0,0,1). Pre-multiplying any quat by
/// this rotates it about the world-Z axis by ``yaw_rad`` without touching
/// pitch or roll.
inline std::array<double, 4> yaw_quat_xyzw(double yaw_rad)
{
  const double half = 0.5 * yaw_rad;
  return { 0.0, 0.0, std::sin(half), std::cos(half) };
}

}  // namespace agi_x2

#endif  // AGI_X2_MATH_UTILS_HPP
