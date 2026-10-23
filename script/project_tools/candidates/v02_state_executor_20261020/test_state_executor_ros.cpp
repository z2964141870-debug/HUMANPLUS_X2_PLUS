#include "state_executor_worker.hpp"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <array>
#include <atomic>
#include <cassert>
#include <iostream>
using namespace std::chrono_literals;
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("state_executor_synthetic_test");
  auto group = node->create_callback_group(rclcpp::CallbackGroupType::Reentrant, false);
  rclcpp::SubscriptionOptions opts; opts.callback_group=group;
  std::array<std::atomic<int>,5> counts{};
  std::array<rclcpp::Subscription<std_msgs::msg::UInt64>::SharedPtr,5> subs;
  std::array<rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr,5> pubs;
  for (int i=0;i<5;++i) {
    auto topic="/plan_a_synthetic/source_"+std::to_string(i);
    subs[i]=node->create_subscription<std_msgs::msg::UInt64>(topic,rclcpp::SensorDataQoS(),
      [&,i](std_msgs::msg::UInt64::SharedPtr){++counts[i];},opts);
    pubs[i]=node->create_publisher<std_msgs::msg::UInt64>(topic,rclcpp::SensorDataQoS());
  }
  std::atomic<bool> blocked{false};
  std::atomic<int> during_block{0};
  auto slow=node->create_wall_timer(20ms,[&]{
    int before=counts[0].load(); blocked=true;
    std::this_thread::sleep_for(80ms); // intentionally exceeds control budget
    during_block += counts[0].load()-before; blocked=false;
  });
  rclcpp::executors::SingleThreadedExecutor state_executor;
  state_executor.add_callback_group(group,node->get_node_base_interface());
  rclcpp::executors::SingleThreadedExecutor main_executor;
  main_executor.add_node(node);
  std::atomic<bool> failed{false}, stop{false};
  agi_x2::StateExecutorWorker<rclcpp::executors::SingleThreadedExecutor> worker(
    state_executor,[&]{failed=true;});
  struct SourceGuard {
    std::atomic<bool>& stop;
    std::thread thread;
    ~SourceGuard() { stop=true; if(thread.joinable()) thread.join(); }
  } source_guard{stop, std::thread([&]{
    std_msgs::msg::UInt64 msg;
    while(!stop) {
      ++msg.data; for(auto& pub:pubs) pub->publish(msg);
      std::this_thread::sleep_for(2ms);
    }
  })};
  auto deadline=std::chrono::steady_clock::now()+3s;
  while(std::chrono::steady_clock::now()<deadline) main_executor.spin_some(100ms);
  // No more callbacks may touch counts after this join, despite live publishers.
  worker.Stop();
  auto after=counts[0].load();
  std::this_thread::sleep_for(10ms);
  bool stopped=(after==counts[0].load());
  stop=true; source_guard.thread.join();
  worker.RethrowIfFailed();
  std::cout<<"received=";
  bool passed=!failed && stopped && during_block>100;
  for(auto& c:counts){std::cout<<c.load()<<",";passed &= c.load()>100;}
  std::cout<<" received_during_80ms_control_blocks="<<during_block
           <<" stopped_before_destroy="<<stopped<<std::endl;
  rclcpp::shutdown();
  return passed?0:1;
}
