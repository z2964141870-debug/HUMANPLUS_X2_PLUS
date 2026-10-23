#pragma once
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

namespace agi_x2 {
// Complete-command publication gate. Integrated with the currently serialized
// deploy writer; independent writer scheduling remains separately gated.
// Lock order: publication gate -> mailbox. No control work holds either lock.
// Publish owns the gate for the ENTIRE hardware-topic batch. Release waits for
// that call to finish and permanently prohibits later Publish invocations.
// This is a process-level guarantee, NOT a DDS queued-delivery guarantee.
template <class Command>
class CommandPublicationGate {
 public:
  struct Envelope {
    Command command;
    int mode;
    bool enabled;
    std::uint64_t generation;
    std::chrono::steady_clock::time_point generated_at;
  };
  bool Commit(Command command, int mode = 0, bool enabled = true) {
    std::lock_guard<std::mutex> publication(publication_);
    if (released_) return false;
    std::lock_guard<std::mutex> mailbox(mailbox_);
    current_=Envelope{std::move(command),mode,enabled,++generation_,std::chrono::steady_clock::now()};
    return true;
  }
  template<class PublishBatch>
  bool Publish(PublishBatch publish_batch) {
    std::lock_guard<std::mutex> publication(publication_);
    if (released_) return false;
    std::optional<Envelope> snapshot;
    {
      std::lock_guard<std::mutex> mailbox(mailbox_);
      snapshot=current_;
    }
    if (!snapshot || !snapshot->enabled) return false;
    // Exceptions propagate; RAII unlocks. Caller MUST treat a throwing batch
    // as partial/failed, never as safe completion or ownership acknowledgement.
    try {
      publish_batch(*snapshot);
    } catch (...) {
      // Latch under the same publication lock BEFORE unwinding. Release
      // racing callback error handling can never acknowledge a failed batch.
      failed_ = true;
      released_ = true;
      current_.reset();
      throw;
    }
    published_generation_ = snapshot->generation;
    return true;
  }
  std::uint64_t PublishedGeneration() {
    std::lock_guard<std::mutex> publication(publication_);
    return published_generation_;
  }
  bool Release() {
    std::lock_guard<std::mutex> publication(publication_);
    released_=true;
    std::lock_guard<std::mutex> mailbox(mailbox_);
    current_.reset();
    return !failed_;
  }
 private:
  std::mutex publication_,mailbox_;
  bool released_=false;
  bool failed_=false;
  std::uint64_t generation_=0;
  std::uint64_t published_generation_=0;
  std::optional<Envelope> current_;
};
} // namespace agi_x2
