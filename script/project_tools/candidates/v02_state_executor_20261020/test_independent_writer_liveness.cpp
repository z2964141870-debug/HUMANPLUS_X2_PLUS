// Offline scheduling experiment. No ROS or robot endpoints exist in this test.
// Demonstrates why independent publishing is NOT hardware-approved yet.
#include "command_publication_gate.hpp"
#include "state_executor_worker.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <future>
#include <thread>
using namespace std::chrono_literals;
struct PollExecutor {
  std::function<void()> step;
  void spin_some(std::chrono::nanoseconds) { step(); }
};
int main() {
  agi_x2::CommandPublicationGate<int> gate;
  std::atomic<int> received{0}, published{0},repeated{0};
  std::promise<void> entered,unblock;
  auto blocked=unblock.get_future();
  std::atomic<bool> control_once{false};
  PollExecutor state{[&]{++received;}};
  PollExecutor writer{[&]{gate.Publish([&](const auto& e){
    assert(e.enabled && e.mode==1 && e.command==42);
    ++published;if(e.generation==1)++repeated;
  });}};
  PollExecutor control{[&]{
    if (!control_once.exchange(true)) {
      assert(gate.Commit(42,1,true));
      entered.set_value();blocked.wait();
    }
  }};
  agi_x2::StateExecutorWorker<PollExecutor> state_worker(state,[]{});
  agi_x2::StateExecutorWorker<PollExecutor> writer_worker(writer,[]{});
  agi_x2::StateExecutorWorker<PollExecutor> control_worker(control,[]{});
  entered.get_future().wait();
  const int state_before=received.load(),writer_before=published.load();
  // Fixed 80ms is test stimulus, NOT an invented safety deadline.
  std::this_thread::sleep_for(80ms);
  assert(received.load()-state_before>10);
  assert(published.load()-writer_before>10);
  assert(repeated.load()>10);
  gate.Release();const int after_ack=published.load();
  unblock.set_value();control_worker.Stop();writer_worker.Stop();state_worker.Stop();
  assert(published==after_ack);
  state_worker.RethrowIfFailed();writer_worker.RethrowIfFailed();control_worker.RethrowIfFailed();
  std::cout<<"PASS: state and writer progress while control blocked; release drains writer\n";
  std::cout<<"KNOWN HARDWARE BLOCKER: fresh states + repeated generation during blocked control. No approved heartbeat/fault policy.\n";
}
