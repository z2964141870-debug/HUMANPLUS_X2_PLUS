#include "state_executor_worker.hpp"
#include <atomic>
#include <cassert>
#include <stdexcept>
#include <iostream>
struct FakeExecutor {
  std::atomic<int> calls{0};
  bool fail = false;
  void spin_some(std::chrono::milliseconds) {
    ++calls;
    if (fail) throw std::runtime_error("injected callback failure");
  }
};
int main() {
  for (int i=0; i<200; ++i) {
    FakeExecutor executor;
    agi_x2::StateExecutorWorker<FakeExecutor> worker(executor, []{});
    worker.Stop(); worker.Stop(); worker.RethrowIfFailed();
  }
  FakeExecutor executor;
  {
    agi_x2::StateExecutorWorker<FakeExecutor> worker(executor, []{});
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  auto stopped = executor.calls.load();
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  assert(stopped > 0 && stopped == executor.calls.load());
  FakeExecutor failing; failing.fail=true;
  std::atomic<bool> notified{false};
  agi_x2::StateExecutorWorker<FakeExecutor> worker(failing, [&]{notified=true;});
  for (int i=0; i<100 && !notified; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  worker.Stop();
  assert(notified);
  bool caught=false;
  try { worker.RethrowIfFailed(); } catch (const std::runtime_error&) {caught=true;}
  assert(caught);
  std::cout << "PASS: 200 immediate starts/stops, idempotent join, no callbacks after teardown, exception propagation\n";
}
