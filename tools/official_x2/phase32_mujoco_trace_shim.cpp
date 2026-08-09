#include "phase32_mujoco_trace_format.h"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

static_assert(sizeof(mjtNum) == sizeof(double),
              "phase32 trace format requires double-precision MuJoCo");

namespace {

using namespace x2_phase32;
using ResetFn = void (*)(const mjModel*, mjData*);
using ForwardFn = void (*)(const mjModel*, mjData*);
using StepFn = void (*)(const mjModel*, mjData*);
using ContactForceFn = void (*)(const mjModel*, const mjData*, int, mjtNum[6]);
using VersionFn = const char* (*)();

struct Mapping {
  int fd = -1;
  std::size_t bytes = 0;
  FileHeader* header = nullptr;
  StateRecord* records = nullptr;
  std::uint32_t max_records = 0;
  double max_time_s = 0.3;
  bool enabled = false;
};

Mapping g_mapping;
std::once_flag g_init_once;
std::atomic<std::uint64_t> g_next_record{0};

template <typename T>
T next_symbol(const char* name) {
  void* symbol = dlsym(RTLD_NEXT, name);
  if (!symbol) {
    std::fprintf(stderr, "[x2-phase32-shim] dlsym(%s) failed: %s\n", name, dlerror());
    std::abort();
  }
  return reinterpret_cast<T>(symbol);
}

std::uint32_t env_u32(const char* name, std::uint32_t fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;
  char* end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);
  if (!end || *end != '\0' || parsed == 0 || parsed > 1000000UL) return fallback;
  return static_cast<std::uint32_t>(parsed);
}

double env_double(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;
  char* end = nullptr;
  double parsed = std::strtod(value, &end);
  return (!end || *end != '\0' || parsed < 0.0) ? fallback : parsed;
}

void initialize_mapping() {
  const char* path = std::getenv("X2_MJ_SHIM_OUTPUT");
  if (!path || !*path) return;
  g_mapping.max_records = env_u32("X2_MJ_SHIM_MAX_RECORDS", 512);
  g_mapping.max_time_s = env_double("X2_MJ_SHIM_MAX_TIME_S", 0.3);
  g_mapping.bytes = sizeof(FileHeader) +
                    static_cast<std::size_t>(g_mapping.max_records) * sizeof(StateRecord);
  g_mapping.fd = ::open(path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (g_mapping.fd < 0) {
    std::fprintf(stderr, "[x2-phase32-shim] open(%s) failed: %s\n", path, std::strerror(errno));
    return;
  }
  if (::ftruncate(g_mapping.fd, static_cast<off_t>(g_mapping.bytes)) != 0) {
    std::fprintf(stderr, "[x2-phase32-shim] ftruncate failed: %s\n", std::strerror(errno));
    ::close(g_mapping.fd);
    g_mapping.fd = -1;
    return;
  }
  void* base = ::mmap(nullptr, g_mapping.bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                      g_mapping.fd, 0);
  if (base == MAP_FAILED) {
    std::fprintf(stderr, "[x2-phase32-shim] mmap failed: %s\n", std::strerror(errno));
    ::close(g_mapping.fd);
    g_mapping.fd = -1;
    return;
  }
  std::memset(base, 0, g_mapping.bytes);
  g_mapping.header = static_cast<FileHeader*>(base);
  g_mapping.records = reinterpret_cast<StateRecord*>(
      static_cast<unsigned char*>(base) + sizeof(FileHeader));
  std::memcpy(g_mapping.header->magic, "X2MJSHIMV1", 11);
  g_mapping.header->format_version = kFormatVersion;
  g_mapping.header->header_size = sizeof(FileHeader);
  g_mapping.header->record_size = sizeof(StateRecord);
  g_mapping.header->max_records = g_mapping.max_records;
  g_mapping.header->process_id = static_cast<std::uint64_t>(::getpid());
  g_mapping.header->max_time_s = g_mapping.max_time_s;
  VersionFn version = next_symbol<VersionFn>("mj_versionString");
  std::snprintf(g_mapping.header->mujoco_version,
                sizeof(g_mapping.header->mujoco_version), "%s", version());
  g_mapping.enabled = true;
}

void flush_mapping() {
  if (!g_mapping.enabled || !g_mapping.header) return;
  ::msync(g_mapping.header, g_mapping.bytes, MS_ASYNC);
}

struct FlushRegistration {
  FlushRegistration() { std::atexit(flush_mapping); }
};

FlushRegistration g_flush_registration;

void copy_values(double* destination, const mjtNum* source, int count, int maximum) {
  const int safe = std::max(0, std::min(count, maximum));
  if (safe > 0 && source) {
    std::memcpy(destination, source, static_cast<std::size_t>(safe) * sizeof(double));
  }
}

void record_state(CallKind kind, const mjModel* model, const mjData* data) {
  std::call_once(g_init_once, initialize_mapping);
  if (!g_mapping.enabled || !model || !data) return;
  if (data->time > g_mapping.max_time_s + 1e-12) return;
  const std::uint64_t index = g_next_record.fetch_add(1, std::memory_order_relaxed);
  if (index >= g_mapping.max_records) {
    __atomic_fetch_add(&g_mapping.header->dropped_records, 1ULL, __ATOMIC_RELAXED);
    return;
  }
  StateRecord* row = &g_mapping.records[index];
  row->committed = 0;
  row->call_kind = static_cast<std::uint32_t>(kind);
  row->sequence = index;
  row->thread_id = static_cast<std::uint64_t>(::syscall(SYS_gettid));
  row->model_pointer = reinterpret_cast<std::uint64_t>(model);
  row->data_pointer = reinterpret_cast<std::uint64_t>(data);
  row->nq = model->nq;
  row->nv = model->nv;
  row->na = model->na;
  row->nu = model->nu;
  row->ncon = data->ncon;
  row->solver_nisland = data->nisland;
  row->solver_nefc = data->nefc;
  row->time_s = data->time;
  copy_values(row->qpos, data->qpos, model->nq, kMaxQ);
  copy_values(row->qvel, data->qvel, model->nv, kMaxV);
  copy_values(row->act, data->act, model->na, kMaxA);
  copy_values(row->ctrl, data->ctrl, model->nu, kMaxU);
  copy_values(row->qacc, data->qacc, model->nv, kMaxV);
  copy_values(row->qacc_warmstart, data->qacc_warmstart, model->nv, kMaxV);
  copy_values(row->qfrc_actuator, data->qfrc_actuator, model->nv, kMaxV);
  copy_values(row->qfrc_constraint, data->qfrc_constraint, model->nv, kMaxV);
  static ContactForceFn contact_force = next_symbol<ContactForceFn>("mj_contactForce");
  const int contacts = std::max(0, std::min(data->ncon, static_cast<int>(kMaxContacts)));
  row->contacts_stored = contacts;
  for (int i = 0; i < contacts; ++i) {
    const mjContact& source = data->contact[i];
    ContactRecord& target = row->contacts[i];
    target.geom1 = source.geom1;
    target.geom2 = source.geom2;
    target.dim = source.dim;
    target.efc_address = source.efc_address;
    target.distance = source.dist;
    std::memcpy(target.position, source.pos, sizeof(target.position));
    std::memcpy(target.frame, source.frame, sizeof(target.frame));
    mjtNum wrench[6] = {0, 0, 0, 0, 0, 0};
    contact_force(model, data, i, wrench);
    std::memcpy(target.wrench, wrench, sizeof(target.wrench));
  }
  __atomic_store_n(&row->committed, 1U, __ATOMIC_RELEASE);
  __atomic_store_n(&g_mapping.header->committed_records, index + 1, __ATOMIC_RELEASE);
}

}  // namespace

extern "C" void mj_resetData(const mjModel* model, mjData* data) {
  static ResetFn original = next_symbol<ResetFn>("mj_resetData");
  original(model, data);
  record_state(kResetPost, model, data);
}

extern "C" void mj_forward(const mjModel* model, mjData* data) {
  static ForwardFn original = next_symbol<ForwardFn>("mj_forward");
  original(model, data);
  record_state(kForwardPost, model, data);
}

extern "C" void mj_step(const mjModel* model, mjData* data) {
  static StepFn original = next_symbol<StepFn>("mj_step");
  original(model, data);
  record_state(kStepPost, model, data);
}

extern "C" std::size_t x2_phase32_file_header_size() {
  return sizeof(x2_phase32::FileHeader);
}

extern "C" std::size_t x2_phase32_state_record_size() {
  return sizeof(x2_phase32::StateRecord);
}
