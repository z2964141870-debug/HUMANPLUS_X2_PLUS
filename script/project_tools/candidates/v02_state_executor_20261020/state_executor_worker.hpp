#pragma once

#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace agi_x2 {
// Owns a state-only executor pump. Start only after callback owners are fully
// constructed. No detached threads or global ROS shutdown on callback failure. Existing CONTROL
// freshness protection remains live on the independent control executor.
// Executor and callback owners MUST outlive this worker.
template <class Executor>
class StateExecutorWorker {
 public:
  explicit StateExecutorWorker(Executor& executor, std::function<void()> on_failure)
      : executor_(executor), thread_([this, on_failure]() {
          try {
            for (;;) {
              {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stop_) return;
              }
              // Bounded work batch. sleep/wakeup avoids busy polling and avoids
              // the cancel-before-spin startup race in Executor::spin().
              executor_.spin_some(std::chrono::milliseconds(2));
              std::unique_lock<std::mutex> lock(mutex_);
              if (wake_.wait_for(lock, std::chrono::milliseconds(1),
                                 [this]() { return stop_; })) return;
            }
          } catch (...) {
            {
              std::lock_guard<std::mutex> lock(mutex_);
              error_ = std::current_exception();
            }
            try { on_failure(); } catch (...) { /* retain original exception */ }
          }
        }) {}
  ~StateExecutorWorker() { Stop(); }
  StateExecutorWorker(const StateExecutorWorker&) = delete;
  StateExecutorWorker& operator=(const StateExecutorWorker&) = delete;

  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
  }
  void RethrowIfFailed() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_) std::rethrow_exception(error_);
  }
 private:
  Executor& executor_;
  std::mutex mutex_;
  std::condition_variable wake_;
  bool stop_ = false;
  std::exception_ptr error_;
  std::thread thread_;  // last: all shared state initialized before start
};
}  // namespace agi_x2
