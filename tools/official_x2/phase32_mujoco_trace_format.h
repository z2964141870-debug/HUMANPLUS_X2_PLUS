#pragma once

#include <cstdint>

namespace x2_phase32 {

constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kMaxQ = 128;
constexpr std::uint32_t kMaxV = 128;
constexpr std::uint32_t kMaxA = 128;
constexpr std::uint32_t kMaxU = 128;
constexpr std::uint32_t kMaxContacts = 64;

enum CallKind : std::uint32_t {
  kResetPost = 1,
  kForwardPost = 2,
  kStepPost = 3,
};

struct FileHeader {
  char magic[16];
  std::uint32_t format_version;
  std::uint32_t header_size;
  std::uint32_t record_size;
  std::uint32_t max_records;
  std::uint64_t committed_records;
  std::uint64_t dropped_records;
  std::uint64_t process_id;
  double max_time_s;
  char mujoco_version[32];
  char reserved[160];
};

struct ContactRecord {
  std::int32_t geom1;
  std::int32_t geom2;
  std::int32_t dim;
  std::int32_t efc_address;
  double distance;
  double position[3];
  double frame[9];
  double wrench[6];
};

struct StateRecord {
  std::uint32_t committed;
  std::uint32_t call_kind;
  std::uint64_t sequence;
  std::uint64_t thread_id;
  std::uint64_t model_pointer;
  std::uint64_t data_pointer;
  std::int32_t nq;
  std::int32_t nv;
  std::int32_t na;
  std::int32_t nu;
  std::int32_t ncon;
  std::int32_t contacts_stored;
  std::int32_t solver_nisland;
  std::int32_t solver_nefc;
  double time_s;
  double qpos[kMaxQ];
  double qvel[kMaxV];
  double act[kMaxA];
  double ctrl[kMaxU];
  double qacc[kMaxV];
  double qacc_warmstart[kMaxV];
  double qfrc_actuator[kMaxV];
  double qfrc_constraint[kMaxV];
  ContactRecord contacts[kMaxContacts];
};

}  // namespace x2_phase32
