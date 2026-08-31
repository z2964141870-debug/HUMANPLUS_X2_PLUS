// Pure-CPU unit test for the X2 obs builders -- no ROS 2, no ONNX runtime.
// Covers the parts most likely to drift away from the Python source of truth:
//   - ProprioceptionBuffer priming + cycling + flatten ordering
//   - StandStillReference / tokenizer obs sizing + ONNX (grouped) layout
//   - Quaternion math used by the gravity / ori-diff blocks
//
// Build standalone (no ROS 2 install needed):
//   cmake -S . -B build -DAGI_X2_OFFLINE_SYNTAX_CHECK=ON
//   cmake --build build
//   ctest --test-dir build --output-on-failure
//
// In a colcon workspace this becomes an ament_cmake_gtest; same code runs.

#include "math_utils.hpp"
#include "policy_parameters.hpp"
#include "proprioception_buffer.hpp"
#include "reference_motion.hpp"
#include "safety.hpp"
#include "startup_ramp.hpp"
#include "tokenizer_obs.hpp"
#include "wrist_bypass.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define EXPECT(cond, msg)                                                   \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " #cond     \
                << "  " << (msg) << "\n";                                   \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

#define EXPECT_NEAR(a, b, tol)                                              \
  do {                                                                      \
    const double _da = (a), _db = (b), _t = (tol);                          \
    if (std::abs(_da - _db) > _t) {                                         \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__                   \
                << "  |" << _da << " - " << _db << "| > " << _t << "\n";    \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

using namespace agi_x2;

static std::array<double, NUM_DOFS> JointVecConst(double v)
{
  std::array<double, NUM_DOFS> a{};
  a.fill(v);
  return a;
}

void TestQuatMath()
{
  // Identity quat -> body gravity should equal world gravity [0,0,-1].
  const std::array<double, 4> id_wxyz = {1.0, 0.0, 0.0, 0.0};
  const auto g = body_frame_gravity_from_quat_wxyz(id_wxyz);
  EXPECT_NEAR(g[0],  0.0, 1e-12);
  EXPECT_NEAR(g[1],  0.0, 1e-12);
  EXPECT_NEAR(g[2], -1.0, 1e-12);

  // 90 deg roll about x: world -z should become body -y.
  const double s = std::sin(M_PI / 4.0);
  const double c = std::cos(M_PI / 4.0);
  const std::array<double, 4> roll90_wxyz = {c, s, 0.0, 0.0};
  const auto g2 = body_frame_gravity_from_quat_wxyz(roll90_wxyz);
  EXPECT_NEAR(g2[0],  0.0, 1e-12);
  EXPECT_NEAR(g2[1], -1.0, 1e-12);
  EXPECT_NEAR(g2[2],  0.0, 1e-12);

  // 6D rot of identity in the columns-flat convention (matches IsaacLab's
  // motion_anchor_ori_b_mf == ``mat[..., :2].reshape(B, -1)``) is
  //     [m[0,0], m[0,1], m[1,0], m[1,1], m[2,0], m[2,1]]
  //   = [   1   ,    0  ,    0  ,    1  ,    0  ,    0  ]
  // (NOT [1,0,0,0,1,0] which is the rows-flat convention -- see
  // math_utils.hpp::rot6d_from_quat_xyzw bug history for the difference and
  // why off-axis the two reps actually disagree.)
  const std::array<double, 4> id_xyzw = {0.0, 0.0, 0.0, 1.0};
  const auto rot6 = rot6d_from_quat_xyzw(id_xyzw);
  EXPECT_NEAR(rot6[0], 1.0, 1e-12);
  EXPECT_NEAR(rot6[1], 0.0, 1e-12);
  EXPECT_NEAR(rot6[2], 0.0, 1e-12);
  EXPECT_NEAR(rot6[3], 1.0, 1e-12);
  EXPECT_NEAR(rot6[4], 0.0, 1e-12);
  EXPECT_NEAR(rot6[5], 0.0, 1e-12);

  // 90 deg yaw about z: m[0,0]=0, m[0,1]=-1, m[1,0]=1, m[1,1]=0, m[2,*]=0.
  // cols-flat: [0, -1, 1, 0, 0, 0]
  const std::array<double, 4> yaw90_xyzw = {0.0, 0.0, s, c};   // (x,y,z,w)
  const auto rot6y = rot6d_from_quat_xyzw(yaw90_xyzw);
  EXPECT_NEAR(rot6y[0],  0.0, 1e-12);
  EXPECT_NEAR(rot6y[1], -1.0, 1e-12);
  EXPECT_NEAR(rot6y[2],  1.0, 1e-12);
  EXPECT_NEAR(rot6y[3],  0.0, 1e-12);
  EXPECT_NEAR(rot6y[4],  0.0, 1e-12);
  EXPECT_NEAR(rot6y[5],  0.0, 1e-12);

  // yaw_from_quat_xyzw / yaw_from_quat_wxyz round-trip with yaw_quat_xyzw.
  // For each test angle, build Rz(theta) and confirm we can recover theta
  // (modulo wrap at +/- pi) from both xyzw and wxyz inputs.
  for (double theta : { -2.5, -0.7, 0.0, 0.4, 1.5, 3.0 }) {
    const auto qx = yaw_quat_xyzw(theta);
    const double recovered_x = yaw_from_quat_xyzw(qx);
    EXPECT_NEAR(recovered_x, theta, 1e-12);
    const auto qw_wxyz = xyzw_to_wxyz(qx);
    const double recovered_w = yaw_from_quat_wxyz(qw_wxyz);
    EXPECT_NEAR(recovered_w, theta, 1e-12);
  }

  // Pure-yaw quat composes correctly: Rz(a) * Rz(b) = Rz(a+b).
  const auto q_a = yaw_quat_xyzw(0.4);
  const auto q_b = yaw_quat_xyzw(0.9);
  const auto q_ab = quat_mul_xyzw(q_a, q_b);
  EXPECT_NEAR(yaw_from_quat_xyzw(q_ab), 1.3, 1e-12);

  std::cout << "  ok TestQuatMath\n";
}

void TestX2PelvisReconstruction()
{
  // Pure pitch makes the expected composition and angular-velocity result
  // transparent: torso pitch = pelvis pitch + waist pitch, and torso pitch
  // rate = pelvis rate + waist rate.
  const auto q_pelvis = x2_pelvis_to_torso_quat_wxyz(0.0, 0.10, 0.0);
  const auto q_waist = x2_pelvis_to_torso_quat_wxyz(0.0, 0.20, 0.0);
  const auto q_torso = quat_mul_wxyz(q_pelvis, q_waist);
  const auto recovered = x2_reconstruct_pelvis_from_torso(
      q_torso, {0.0, 0.70, 0.0},
      0.0, 0.20, 0.0,
      0.0, 0.30, 0.0);
  const auto m = rotation_matrix_xyzw(wxyz_to_xyzw(recovered.quat_wxyz));
  EXPECT_NEAR(std::asin(-m[6]), 0.10, 1e-12);
  EXPECT_NEAR(recovered.ang_vel[0], 0.0, 1e-12);
  EXPECT_NEAR(recovered.ang_vel[1], 0.40, 1e-12);
  EXPECT_NEAR(recovered.ang_vel[2], 0.0, 1e-12);

  // Captured X2 sample from the 2026-08-28 supported probe. Raw torso pitch
  // was +11.01 deg while waist pitch was +12.83 deg; reconstructed pelvis
  // must be near -1.81 deg, not +11 deg.
  const auto captured = x2_reconstruct_pelvis_from_torso(
      {-0.995255, -0.007246, -0.095905, 0.014755},
      {-0.000718, -0.002289, 0.002662},
      0.007190, 0.223904, 0.092608,
      0.0, 0.0, 0.0);
  const auto cm = rotation_matrix_xyzw(wxyz_to_xyzw(captured.quat_wxyz));
  EXPECT_NEAR(std::asin(-cm[6]) * 180.0 / M_PI, -1.81, 0.05);
  std::cout << "  ok TestX2PelvisReconstruction\n";
}

namespace {

/// Write a synthetic X2M2 motion file to ``path`` with a single frame whose
/// joint pose = default_angles and whose root quaternion has yaw =
/// ``yaw_rad`` (no pitch/roll). Returns true on success. Used by the
/// yaw-anchor test below; matches the file format documented in
/// reference_motion.hpp.
bool WriteSyntheticMotionFile(const std::string& path,
                              double             yaw_rad,
                              std::uint32_t      num_frames = 2)
{
  constexpr std::uint32_t kMagic = 0x58324D32u;  // "X2M2"
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return false;

  const std::uint32_t num_dofs = NUM_DOFS;
  const double        fps      = 30.0;
  f.write(reinterpret_cast<const char*>(&kMagic),      sizeof(kMagic));
  f.write(reinterpret_cast<const char*>(&num_frames),  sizeof(num_frames));
  f.write(reinterpret_cast<const char*>(&num_dofs),    sizeof(num_dofs));
  f.write(reinterpret_cast<const char*>(&fps),         sizeof(fps));

  const auto qx = yaw_quat_xyzw(yaw_rad);
  for (std::uint32_t i = 0; i < num_frames; ++i) {
    for (std::size_t d = 0; d < NUM_DOFS; ++d) {
      const double v = default_angles[d];
      f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }
    f.write(reinterpret_cast<const char*>(qx.data()), sizeof(double) * 4);
  }
  return static_cast<bool>(f);
}

}  // namespace

void TestPklMotionYawAnchor()
{
  // Write a 2-frame motion clip whose recorded yaw is +90 deg (in some
  // arbitrary world frame the recording was captured in). Load it; verify
  // pre-Anchor Sample() returns that yaw verbatim, then Anchor it to a
  // robot quaternion at yaw = -45 deg and verify Sample() now returns
  // pure-yaw rotations matching the robot's heading. This simulates the
  // exact production path: deploy harness loads a pkl, sees the IMU yaw
  // doesn't match the file's frame-0 yaw, and re-anchors so the policy's
  // motion-anchor diff at t=0 is identity.
  const std::string path = "/tmp/agi_x2_yaw_anchor_test.x2m2";
  EXPECT(WriteSyntheticMotionFile(path, /*yaw_rad=*/M_PI / 2.0),
         "failed to write synthetic motion file");

  auto ref = PklMotionReference::Load(path);

  // Pre-anchor: identity anchor leaves the recorded yaw alone.
  EXPECT_NEAR(ref->yaw_anchor_delta(), 0.0, 1e-12);
  const auto pre = ref->Sample(0.0);
  EXPECT_NEAR(yaw_from_quat_xyzw(pre.root_quat_xyzw),  M_PI / 2.0, 1e-12);

  // Anchor to a robot at yaw = -45 deg (wxyz convention from IMU).
  const double robot_yaw = -M_PI / 4.0;
  const auto   robot_q   = xyzw_to_wxyz(yaw_quat_xyzw(robot_yaw));
  ref->Anchor(robot_q);

  // The applied delta should be (robot_yaw - motion_yaw) = -3pi/4.
  EXPECT_NEAR(ref->yaw_anchor_delta(), -3.0 * M_PI / 4.0, 1e-12);

  // Post-anchor: every served frame's root yaw should equal robot_yaw.
  // (Looped playback is fine -- both frames had the same yaw.)
  for (double t : { 0.0, 0.1, 1.7, 5.5 }) {
    const auto frame = ref->Sample(t);
    EXPECT_NEAR(yaw_from_quat_xyzw(frame.root_quat_xyzw), robot_yaw, 1e-12);
  }

  std::remove(path.c_str());
  std::cout << "  ok TestPklMotionYawAnchor\n";
}

void TestProprioceptionPriming()
{
  ProprioceptionBuffer pb;
  EXPECT(!pb.IsPrimed(), "buffer should not be primed before first append");

  // First append should broadcast-fill all 10 history slots with `1.0` for
  // every term -> entire 990-D output is 1.0.
  const auto jvec = JointVecConst(1.0);
  pb.Append({1.0, 1.0, 1.0}, jvec, jvec, jvec, {1.0, 1.0, 1.0});
  EXPECT(pb.IsPrimed(), "buffer should be primed after first append");

  const auto flat = pb.GetFlat();
  EXPECT(flat.size() == PROP_DIM, "flatten size != 990");
  for (std::size_t i = 0; i < flat.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(flat[i]), 1.0, 1e-12);
  }
  std::cout << "  ok TestProprioceptionPriming\n";
}

void TestProprioceptionTermOrderAndAging()
{
  // Push 11 distinct samples; the oldest should fall off and the newest
  // should be at the END of each term's 10-frame block.
  ProprioceptionBuffer pb;
  for (int k = 0; k < 11; ++k) {
    const double v = static_cast<double>(k);
    pb.Append({v, v, v},
              JointVecConst(v), JointVecConst(v), JointVecConst(v),
              {v, v, v});
  }
  const auto flat = pb.GetFlat();

  // Term 0: base_ang_vel, 10 frames * 3 dim = 30. After 11 appends the
  // oldest retained sample is value=1.0 (since k=0 fell off), and the
  // newest is 10.0.
  EXPECT_NEAR(static_cast<double>(flat[0]),  1.0, 1e-12);   // oldest, slot 0
  EXPECT_NEAR(static_cast<double>(flat[27]), 10.0, 1e-12);  // newest, slot 9
  EXPECT_NEAR(static_cast<double>(flat[28]), 10.0, 1e-12);
  EXPECT_NEAR(static_cast<double>(flat[29]), 10.0, 1e-12);

  // Term 1 starts at offset 30 (joint_pos_rel, 31-D x 10).
  // Oldest slot, joint 0 of frame 0 = 1.0
  EXPECT_NEAR(static_cast<double>(flat[30]), 1.0, 1e-12);
  // Newest slot starts at 30 + 9 * 31 = 309
  EXPECT_NEAR(static_cast<double>(flat[309]), 10.0, 1e-12);

  // Term 4 (gravity_dir) is the LAST 30 floats, so newest sample at
  // indices 987..989 should be 10.0.
  EXPECT_NEAR(static_cast<double>(flat[987]), 10.0, 1e-12);
  EXPECT_NEAR(static_cast<double>(flat[989]), 10.0, 1e-12);

  std::cout << "  ok TestProprioceptionTermOrderAndAging\n";
}

void TestStandStillTokenizerSize()
{
  StandStillReference ref;
  const std::array<double, 4> id_wxyz = {1.0, 0.0, 0.0, 0.0};
  const auto tok = BuildTokenizerObs(ref, /*current_time=*/0.0, id_wxyz);
  EXPECT(tok.size() == TOK_DIM, "tokenizer size != 680");

  // ONNX layout for the 680-D tokenizer slice mirrors IsaacLab's buggy
  // `command_multi_future_nonflat` reshape (see tokenizer_obs.hpp file
  // header for the full bug-history rationale). Concretely, 10 rows of 68
  // floats flattened row-major:
  //
  //   row k < 5:   [jpos_f(2k)(31)     | jpos_f(2k+1)(31)     | ori_fk(6)]
  //   row k >= 5:  [jvel_f(2(k-5))(31) | jvel_f(2(k-5)+1)(31) | ori_fk(6)]
  //
  // For a StandStillReference at any time t every jpos future frame equals
  // default_angles permuted into IL order, every jvel future frame is zero,
  // and every ori future frame (cur=identity, future=identity) is the 6D
  // representation of identity = [1,0,0, 1,0,0] (first two columns of I3
  // flattened row-major, see math_utils::rot6d_from_quat_xyzw).
  //
  // So rows 0..4 contain default_angles in BOTH halves (jpos pair), rows
  // 5..9 contain zeros in BOTH halves (jvel pair), and every row's last 6
  // floats are the identity rot6.
  //
  // Tolerance is 1e-6 because the tokenizer obs is float32 (the ONNX input
  // dtype) but default_angles[] is double, so we expect <= 1 ULP of float
  // quantization noise on jpos. jvel/ori are exact-zero/exact-{0,1} in
  // float32 so use 1e-12.

  for (std::size_t k = 0; k < 10; ++k) {
    const std::size_t base = k * 68;  // each row is 62 (cmd) + 6 (ori)

    if (k < 5) {
      // Two jpos halves: each is default_angles in IL order.
      for (std::size_t half = 0; half < 2; ++half) {
        const std::size_t hbase = base + half * NUM_DOFS;
        for (std::size_t il = 0; il < NUM_DOFS; ++il) {
          const std::size_t mj = static_cast<std::size_t>(isaaclab_to_mujoco[il]);
          EXPECT_NEAR(static_cast<double>(tok[hbase + il]),
                      default_angles[mj], 1e-6);
        }
      }
    } else {
      // Two jvel halves: each is zero.
      for (std::size_t i = 0; i < 2 * NUM_DOFS; ++i) {
        EXPECT_NEAR(static_cast<double>(tok[base + i]), 0.0, 1e-12);
      }
    }

    // ori block (offset +62 within row): identity rotation, first two
    // columns of I3 flattened row-major = [1, 0, 0, 1, 0, 0]
    const std::size_t obase = base + 62;
    EXPECT_NEAR(static_cast<double>(tok[obase + 0]), 1.0, 1e-7);
    EXPECT_NEAR(static_cast<double>(tok[obase + 1]), 0.0, 1e-7);
    EXPECT_NEAR(static_cast<double>(tok[obase + 2]), 0.0, 1e-7);
    EXPECT_NEAR(static_cast<double>(tok[obase + 3]), 1.0, 1e-7);
    EXPECT_NEAR(static_cast<double>(tok[obase + 4]), 0.0, 1e-7);
    EXPECT_NEAR(static_cast<double>(tok[obase + 5]), 0.0, 1e-7);
  }

  const double robot_yaw = 0.7;
  ref.Anchor(xyzw_to_wxyz(yaw_quat_xyzw(robot_yaw)));
  EXPECT_NEAR(ref.yaw_anchor_delta(), robot_yaw, 1e-12);
  EXPECT_NEAR(
      yaw_from_quat_xyzw(ref.Sample(0.0).root_quat_xyzw),
      robot_yaw, 1e-12);

  std::cout << "  ok TestStandStillTokenizerSize\n";
}

// ---------------------------------------------------------------------------
// Wrist bypass: pin the override invariants documented in
// include/wrist_bypass.hpp -- the right slots, only those slots, and a
// truthful max-delta return value for the periodic status line.
// ---------------------------------------------------------------------------
void TestWristBypassOverridesExactly4Slots()
{
  // Arrange: per-DOF target_pos_mj filled with a sentinel value (1.234)
  // so we can detect any accidental writes outside the 4 wrist slots,
  // and a synthetic ReferenceFrame whose joint_pos_mj has unique values
  // per slot (slot index * 0.01 rad) so we can assert the right value
  // landed in the right place after the override.
  std::array<double, NUM_DOFS> target_pos_mj;
  target_pos_mj.fill(1.234);

  ReferenceFrame ref{};
  for (std::size_t mj = 0; mj < NUM_DOFS; ++mj) {
    ref.joint_pos_mj[mj] = static_cast<double>(mj) * 0.01;
    ref.joint_vel_mj[mj] = 0.0;
  }
  ref.root_quat_xyzw = {0.0, 0.0, 0.0, 1.0};

  // Act.
  const double max_delta = ApplyWristBypass(target_pos_mj, ref);

  // Assert (1): the bypassed indices match the documented set.
  EXPECT(kBypassedWristMjDofs.size() == 4,
         "kBypassedWristMjDofs must have exactly 4 entries");
  EXPECT(kBypassedWristMjDofs[0] == 20, "slot 0 must be left_wrist_pitch (MJ 20)");
  EXPECT(kBypassedWristMjDofs[1] == 21, "slot 1 must be left_wrist_roll  (MJ 21)");
  EXPECT(kBypassedWristMjDofs[2] == 27, "slot 2 must be right_wrist_pitch(MJ 27)");
  EXPECT(kBypassedWristMjDofs[3] == 28, "slot 3 must be right_wrist_roll (MJ 28)");

  // Assert (2): all four bypassed slots now carry the IK reference value;
  // every other slot is untouched (== sentinel).
  for (std::size_t mj = 0; mj < NUM_DOFS; ++mj) {
    const bool is_bypassed = (mj == 20 || mj == 21 || mj == 27 || mj == 28);
    if (is_bypassed) {
      EXPECT_NEAR(target_pos_mj[mj], ref.joint_pos_mj[mj], 1e-12);
    } else {
      EXPECT_NEAR(target_pos_mj[mj], 1.234, 1e-12);
    }
  }

  // Assert (3): max_delta is the largest |sentinel - ref| across the
  // bypassed slots. With sentinel=1.234 and ref values {0.20, 0.21,
  // 0.27, 0.28} the worst case is slot 20: |1.234 - 0.20| = 1.034.
  EXPECT_NEAR(max_delta, 1.234 - 0.20, 1e-12);

  std::cout << "  ok TestWristBypassOverridesExactly4Slots\n";
}

// ---------------------------------------------------------------------------
// Wrist bypass: when the IK reference matches the policy target exactly,
// max_delta must be zero and target_pos_mj must remain bit-identical.
// Catches a silly bug where an off-by-one or a wrong subtraction direction
// would still happen to overwrite the right slots but report nonsense in
// the periodic status line.
// ---------------------------------------------------------------------------
void TestWristBypassZeroDeltaWhenAligned()
{
  std::array<double, NUM_DOFS> target_pos_mj;
  for (std::size_t mj = 0; mj < NUM_DOFS; ++mj) {
    target_pos_mj[mj] = static_cast<double>(mj) * 0.05;
  }
  std::array<double, NUM_DOFS> snapshot = target_pos_mj;

  ReferenceFrame ref{};
  ref.joint_pos_mj = target_pos_mj;  // same values: nothing to override
  ref.joint_vel_mj.fill(0.0);
  ref.root_quat_xyzw = {0.0, 0.0, 0.0, 1.0};

  const double max_delta = ApplyWristBypass(target_pos_mj, ref);

  EXPECT_NEAR(max_delta, 0.0, 1e-15);
  for (std::size_t mj = 0; mj < NUM_DOFS; ++mj) {
    EXPECT_NEAR(target_pos_mj[mj], snapshot[mj], 1e-15);
  }
  std::cout << "  ok TestWristBypassZeroDeltaWhenAligned\n";
}

void TestWristFreezeOverridesExactly6Slots()
{
  std::array<double, NUM_DOFS> target_pos_mj;
  target_pos_mj.fill(1.234);
  const double max_delta = ApplyWristFreeze(target_pos_mj);

  EXPECT(kFrozenWristMjDofs.size() == 6,
         "kFrozenWristMjDofs must have exactly 6 entries");
  for (std::size_t mj = 0; mj < NUM_DOFS; ++mj) {
    const bool frozen =
        (mj == 19 || mj == 20 || mj == 21 || mj == 26 || mj == 27 || mj == 28);
    EXPECT_NEAR(target_pos_mj[mj], frozen ? default_angles[mj] : 1.234, 1e-12);
  }
  EXPECT_NEAR(max_delta, 1.234, 1e-12);
  std::cout << "  ok TestWristFreezeOverridesExactly6Slots\n";
}

void TestSuspendedStartupRamp()
{
  EXPECT_NEAR(StartupSmoothStep(-1.0, 3.0), 0.0, 1e-15);
  EXPECT_NEAR(StartupSmoothStep(0.0, 3.0), 0.0, 1e-15);
  EXPECT_NEAR(StartupSmoothStep(1.5, 3.0), 0.5, 1e-15);
  EXPECT_NEAR(StartupSmoothStep(3.0, 3.0), 1.0, 1e-15);
  EXPECT_NEAR(StartupSmoothStep(4.0, 3.0), 1.0, 1e-15);

  auto anchor = default_angles;
  anchor[0] += 0.6;
  anchor[10] -= 0.3;
  EXPECT_NEAR(StartupPoseRampDuration(anchor, 2.0, 0.15), 4.0, 1e-12);

  const auto begin = StartupPoseTarget(anchor, 0.0);
  const auto middle = StartupPoseTarget(anchor, 0.5);
  const auto end = StartupPoseTarget(anchor, 1.0);
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    EXPECT_NEAR(begin[i], anchor[i], 1e-15);
    EXPECT_NEAR(middle[i], 0.5 * (anchor[i] + default_angles[i]), 1e-15);
    EXPECT_NEAR(end[i], default_angles[i], 1e-15);
  }

  const auto desired = JointVecConst(1.0);
  const auto custom_anchor = JointVecConst(-1.0);
  EXPECT_NEAR(
      StartupPoseRampDuration(custom_anchor, desired, 2.0, 0.5),
      4.0, 1e-12);
  const auto custom_middle = StartupPoseTarget(
      custom_anchor, desired, 0.5);
  for (double value : custom_middle) {
    EXPECT_NEAR(value, 0.0, 1e-15);
  }
  std::cout << "  ok TestSuspendedStartupRamp\n";
}

void TestSupportedPolicyProbeTargets()
{
  const auto static_target = JointVecConst(1.0);
  auto policy_target = JointVecConst(2.0);
  policy_target[1] = 0.0;

  const auto half = SupportedPolicyTarget(static_target, policy_target, 0.5);
  EXPECT_NEAR(half[0], 1.01, 1e-15);
  EXPECT_NEAR(half[1], 0.99, 1e-15);
  EXPECT_NEAR(half[12], 1.01, 1e-15);
  EXPECT_NEAR(half[15], 1.015, 1e-15);
  EXPECT_NEAR(half[30], 1.015, 1e-15);

  const auto full = SupportedPolicyTarget(static_target, policy_target, 1.0);
  EXPECT_NEAR(full[11], 1.02, 1e-15);
  EXPECT_NEAR(full[14], 1.02, 1e-15);
  EXPECT_NEAR(full[28], 1.03, 1e-15);
  EXPECT_NEAR(full[30], 1.03, 1e-15);

  const auto returned = SupportedPolicyReturnTarget(full, static_target, 1.0);
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    EXPECT_NEAR(returned[i], static_target[i], 1e-15);
  }

  const auto staged = SupportedPolicyTarget(
      static_target, policy_target, 1.0, 0.08, 0.06, 0.10, 0.05);
  EXPECT_NEAR(staged[11], 1.08, 1e-15);
  EXPECT_NEAR(staged[14], 1.06, 1e-15);
  EXPECT_NEAR(staged[28], 1.10, 1e-15);
  EXPECT_NEAR(staged[30], 1.05, 1e-15);

  auto default_centered_policy = default_angles;
  default_centered_policy[0] += 1.0;
  default_centered_policy[12] -= 1.0;
  default_centered_policy[15] += 1.0;
  default_centered_policy[30] -= 1.0;
  const auto default_centered_start = SupportedPolicyTargetAroundDefault(
      static_target, default_centered_policy, 0.0, 0.60, 0.20, 0.25, 0.08);
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    EXPECT_NEAR(default_centered_start[i], static_target[i], 1e-15);
  }
  const auto default_centered_full = SupportedPolicyTargetAroundDefault(
      static_target, default_centered_policy, 1.0, 0.60, 0.20, 0.25, 0.08);
  EXPECT_NEAR(default_centered_full[0], default_angles[0] + 0.60, 1e-15);
  EXPECT_NEAR(default_centered_full[12], default_angles[12] - 0.20, 1e-15);
  EXPECT_NEAR(default_centered_full[15], default_angles[15] + 0.25, 1e-15);
  EXPECT_NEAR(default_centered_full[30], default_angles[30] - 0.08, 1e-15);

  const auto rate_limited = SlewLimitTarget(
      static_target, policy_target, 0.15, 0.02);
  EXPECT_NEAR(rate_limited[0], 1.003, 1e-15);
  EXPECT_NEAR(rate_limited[1], 0.997, 1e-15);
  const auto rate_disabled = SlewLimitTarget(
      static_target, policy_target, 0.0, 0.02);
  EXPECT_NEAR(rate_disabled[0], policy_target[0], 1e-15);

  auto applied_target = default_angles;
  for (std::size_t mj = 0; mj < NUM_DOFS; ++mj) {
    const std::size_t il = static_cast<std::size_t>(mujoco_to_isaaclab[mj]);
    applied_target[mj] += x2_action_scale[mj] * (0.01 * (il + 1));
  }
  const auto applied_action = AppliedActionFromTarget(applied_target);
  for (std::size_t il = 0; il < NUM_DOFS; ++il) {
    EXPECT_NEAR(applied_action[il], 0.01 * (il + 1), 1e-12);
  }
  std::cout << "  ok TestSupportedPolicyProbeTargets\n";
}

void TestCheckpointActionScaleContract()
{
  // models/x2_sonic_frozen_g1core_lora_v2.onnx.phi.json stores X2 scales in
  // IsaacLab order. This explicit checkpoint contract catches a self-
  // consistent but wrong C++ table (an inverse round-trip alone cannot).
  const std::array<double, NUM_DOFS> expected_il = {
      0.3027293234752049, 0.3027293234752049, 0.7466542707039959,
      0.3027293234752049, 0.3027293234752049, 0.6315513320135467,
      0.3027293234752049, 0.3027293234752049, 0.6315513320135467,
      0.3027293234752049, 0.3027293234752049, 0.03874045256912915,
      0.6315513320135467, 0.6315513320135467, 0.6315513320135467,
      0.6315513320135467, 0.008940104439029803, 0.6315513320135467,
      0.6315513320135467, 0.4210342213423645, 0.4210342213423645,
      0.4210342213423645, 0.4210342213423645, 0.4210342213423645,
      0.4210342213423645, 0.4210342213423645, 0.4210342213423645,
      0.08940104439029803, 0.08940104439029803,
      0.08940104439029803, 0.08940104439029803,
  };
  for (std::size_t mj = 0; mj < NUM_DOFS; ++mj) {
    const std::size_t il = static_cast<std::size_t>(mujoco_to_isaaclab[mj]);
    EXPECT_NEAR(x2_action_scale[mj], expected_il[il], 1e-9);
  }
  std::cout << "  ok TestCheckpointActionScaleContract\n";
}

void TestThermalMonitorIgnoresUnavailableHeadTelemetry()
{
  ThermalMonitor monitor;
  auto coil = JointVecConst(30.0);
  auto motor = JointVecConst(25.0);
  coil[29] = 0.0;
  motor[29] = 148.0;

  EXPECT(monitor.Update(coil, motor, 0.0).empty(),
         "invalid X2 head temperature must not trigger HOT");
  EXPECT(!monitor.AnyHot(), "invalid head telemetry must stay non-hot");

  motor[0] = 95.0;
  EXPECT(!monitor.Update(coil, motor, 1.0).empty(),
         "valid body temperature should trigger HOT");
  EXPECT(monitor.HotCount() == 1, "exactly one body motor should be hot");
  EXPECT(monitor.Update(coil, motor, 2.0).empty(),
         "persistent HOT state should not retrigger before reminder");

  motor[0] = 80.0;
  EXPECT(!monitor.Update(coil, motor, 3.0).empty(),
         "valid body cooling should be reported");
  EXPECT(!monitor.AnyHot(), "cooled body motor should clear HOT state");
  std::cout << "  ok TestThermalMonitorIgnoresUnavailableHeadTelemetry\n";
}

int main()
{
  std::cout << "agi_x2_deploy_onnx_ref unit tests\n";
  TestQuatMath();
  TestX2PelvisReconstruction();
  TestProprioceptionPriming();
  TestProprioceptionTermOrderAndAging();
  TestStandStillTokenizerSize();
  TestPklMotionYawAnchor();
  TestWristBypassOverridesExactly4Slots();
  TestWristBypassZeroDeltaWhenAligned();
  TestWristFreezeOverridesExactly6Slots();
  TestSuspendedStartupRamp();
  TestSupportedPolicyProbeTargets();
  TestCheckpointActionScaleContract();
  TestThermalMonitorIgnoresUnavailableHeadTelemetry();
  std::cout << "all OK\n";
  return 0;
}
