#include "deploy_logger.hpp"

#include <filesystem>
#include <iomanip>
#include <stdexcept>

namespace agi_x2 {

namespace fs = std::filesystem;

namespace {

void WriteHeaderJointVec(std::ofstream& f, const char* prefix)
{
  f << "t";
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    f << "," << prefix << "_" << mujoco_joint_names[i];
  }
  f << "\n";
}

void WriteRow(std::ofstream& f, double t,
              const std::array<double, NUM_DOFS>& v)
{
  f << std::fixed << std::setprecision(6) << t;
  for (double x : v) f << "," << x;
  f << "\n";
}

}  // namespace

DeployLogger::DeployLogger(const std::string& output_dir, bool enabled)
    : enabled_(enabled),
      output_dir_(output_dir)
{
  if (!enabled_) return;

  fs::create_directories(output_dir_);

  tick_.open(output_dir_ + "/tick.csv");
  tick_ << "t,ramp_alpha,dry_run,tilt_trip,reason\n";

  policy_target_pos_.open(output_dir_ + "/policy_target_pos.csv");
  WriteHeaderJointVec(policy_target_pos_, "policy_target");

  target_pos_.open(output_dir_ + "/target_pos.csv");
  WriteHeaderJointVec(target_pos_, "target");

  joint_pos_.open(output_dir_ + "/joint_pos.csv");
  WriteHeaderJointVec(joint_pos_, "q");

  joint_vel_.open(output_dir_ + "/joint_vel.csv");
  WriteHeaderJointVec(joint_vel_, "dq");

  tracking_error_.open(output_dir_ + "/tracking_error.csv");
  WriteHeaderJointVec(tracking_error_, "target_minus_q");

  joint_effort_.open(output_dir_ + "/joint_effort.csv");
  WriteHeaderJointVec(joint_effort_, "measured_tau_nm");

  stiffness_.open(output_dir_ + "/stiffness.csv");
  WriteHeaderJointVec(stiffness_, "kp");

  damping_.open(output_dir_ + "/damping.csv");
  WriteHeaderJointVec(damping_, "kd");

  pd_torque_.open(output_dir_ + "/pd_torque.csv");
  WriteHeaderJointVec(pd_torque_, "estimated_tau_nm");

  coil_temp_.open(output_dir_ + "/coil_temp.csv");
  WriteHeaderJointVec(coil_temp_, "coil_temp_c");

  motor_temp_.open(output_dir_ + "/motor_temp.csv");
  WriteHeaderJointVec(motor_temp_, "motor_temp_c");

  motor_voltage_.open(output_dir_ + "/motor_voltage.csv");
  WriteHeaderJointVec(motor_voltage_, "motor_voltage_v");

  domain_state_.open(output_dir_ + "/domain_state.csv");
  domain_state_ << "t,leg,waist,arm,head\n";

  action_il_.open(output_dir_ + "/action_il.csv");
  // action_il is in IL order, so we don't tag with mj joint names; just
  // index it. Anyone post-processing can re-permute via mujoco_to_isaaclab.
  action_il_ << "t";
  for (std::size_t i = 0; i < NUM_DOFS; ++i) action_il_ << ",a_il_" << i;
  action_il_ << "\n";

  imu_.open(output_dir_ + "/imu.csv");
  imu_ << "t,qw,qx,qy,qz,wx,wy,wz,raw_torso_qw,raw_torso_qx,"
          "raw_torso_qy,raw_torso_qz,raw_torso_wx,raw_torso_wy,"
          "raw_torso_wz\n";

  if (!tick_ || !policy_target_pos_ || !target_pos_ || !joint_pos_
      || !joint_vel_ || !tracking_error_ || !joint_effort_ || !stiffness_
      || !damping_ || !pd_torque_ || !coil_temp_ || !motor_temp_
      || !motor_voltage_ || !domain_state_ || !action_il_ || !imu_) {
    throw std::runtime_error(
        "DeployLogger: failed to open one or more CSV files in "
        + output_dir_);
  }

  // Make a successfully-created logger visible immediately. Besides
  // preserving headers across a hard power cut, this distinguishes a logger
  // that never received Log() calls from one that could not open its files.
  tick_.flush();
  policy_target_pos_.flush();
  target_pos_.flush();
  joint_pos_.flush();
  joint_vel_.flush();
  tracking_error_.flush();
  joint_effort_.flush();
  stiffness_.flush();
  damping_.flush();
  pd_torque_.flush();
  coil_temp_.flush();
  motor_temp_.flush();
  motor_voltage_.flush();
  domain_state_.flush();
  action_il_.flush();
  imu_.flush();
}

DeployLogger::~DeployLogger()
{
  if (!enabled_) return;
  tick_.flush();
  policy_target_pos_.flush();
  target_pos_.flush();
  joint_pos_.flush();
  joint_vel_.flush();
  tracking_error_.flush();
  joint_effort_.flush();
  stiffness_.flush();
  damping_.flush();
  pd_torque_.flush();
  coil_temp_.flush();
  motor_temp_.flush();
  motor_voltage_.flush();
  domain_state_.flush();
  action_il_.flush();
  imu_.flush();
}

void DeployLogger::Log(double                              now_s,
                       const RobotState&                   robot_state,
                       const std::array<double, NUM_DOFS>& policy_target_pos_mj,
                       const std::array<double, NUM_DOFS>& action_il,
                       const SafeCommand&                  safe_cmd)
{
  if (!enabled_) return;
  std::lock_guard<std::mutex> lk(io_mutex_);

  std::array<double, NUM_DOFS> tracking_error{};
  std::array<double, NUM_DOFS> pd_torque{};
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    tracking_error[i] =
        safe_cmd.target_pos_mj[i] - robot_state.joint_pos_mj[i];
    // The deploy publishes a zero target velocity, so this is the exact
    // requested PD term before any motor-side saturation or protection.
    pd_torque[i] = safe_cmd.stiffness_mj[i] * tracking_error[i]
                 - safe_cmd.damping_mj[i] * robot_state.joint_vel_mj[i];
  }

  tick_ << std::fixed << std::setprecision(6) << now_s
        << "," << safe_cmd.ramp_alpha
        << "," << (safe_cmd.dry_run ? 1 : 0)
        << "," << (safe_cmd.tilt_trip ? 1 : 0)
        << ",\"" << safe_cmd.reason << "\"\n";

  WriteRow(policy_target_pos_, now_s, policy_target_pos_mj);
  WriteRow(target_pos_,        now_s, safe_cmd.target_pos_mj);
  WriteRow(joint_pos_,         now_s, robot_state.joint_pos_mj);
  WriteRow(joint_vel_,         now_s, robot_state.joint_vel_mj);
  WriteRow(tracking_error_,    now_s, tracking_error);
  WriteRow(joint_effort_,      now_s, robot_state.joint_effort_nm);
  WriteRow(stiffness_,         now_s, safe_cmd.stiffness_mj);
  WriteRow(damping_,           now_s, safe_cmd.damping_mj);
  WriteRow(pd_torque_,         now_s, pd_torque);
  WriteRow(coil_temp_,         now_s, robot_state.coil_temp_c);
  WriteRow(motor_temp_,        now_s, robot_state.motor_temp_c);
  WriteRow(motor_voltage_,     now_s, robot_state.motor_voltage_v);

  domain_state_ << std::fixed << std::setprecision(6) << now_s;
  for (std::uint8_t value : robot_state.domain_state) {
    domain_state_ << "," << static_cast<unsigned int>(value);
  }
  domain_state_ << "\n";

  // action_il written manually because it has its own column naming above.
  action_il_ << std::fixed << std::setprecision(6) << now_s;
  for (double x : action_il) action_il_ << "," << x;
  action_il_ << "\n";

  imu_ << std::fixed << std::setprecision(6) << now_s
       << "," << robot_state.base_quat_wxyz[0]
       << "," << robot_state.base_quat_wxyz[1]
       << "," << robot_state.base_quat_wxyz[2]
       << "," << robot_state.base_quat_wxyz[3]
       << "," << robot_state.base_ang_vel[0]
       << "," << robot_state.base_ang_vel[1]
       << "," << robot_state.base_ang_vel[2]
       << "," << robot_state.raw_torso_quat_wxyz[0]
       << "," << robot_state.raw_torso_quat_wxyz[1]
       << "," << robot_state.raw_torso_quat_wxyz[2]
       << "," << robot_state.raw_torso_quat_wxyz[3]
       << "," << robot_state.raw_torso_ang_vel[0]
       << "," << robot_state.raw_torso_ang_vel[1]
       << "," << robot_state.raw_torso_ang_vel[2]
       << "\n";

  // Flush all streams once a second. 2026-08-04 incident postmortem had
  // ZERO rows — the operator pulled the battery ~14 s into the session
  // and every CSV was still sitting in ofstream buffers. A 1 Hz flush
  // costs nothing at this data rate and guarantees a power-cut leaves
  // at most the last second unrecorded.
  if (now_s - last_flush_s_ >= 1.0) {
    last_flush_s_ = now_s;
    tick_.flush();
    policy_target_pos_.flush();
    target_pos_.flush();
    joint_pos_.flush();
    joint_vel_.flush();
    tracking_error_.flush();
    joint_effort_.flush();
    stiffness_.flush();
    damping_.flush();
    pd_torque_.flush();
    coil_temp_.flush();
    motor_temp_.flush();
    motor_voltage_.flush();
    domain_state_.flush();
    action_il_.flush();
    imu_.flush();
  }
}

}  // namespace agi_x2
