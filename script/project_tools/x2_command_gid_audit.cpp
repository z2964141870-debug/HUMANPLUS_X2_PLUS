// Subscription-only audit for Humble builds without Python MessageInfo GIDs.
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <aimdk_msgs/msg/joint_command_array.hpp>

template <class Bytes>
std::string Hex(const Bytes& bytes) {
  std::ostringstream out;
  for (auto value : bytes) {
    out << std::hex << std::setfill('0') << std::setw(2)
        << static_cast<unsigned>(value);
  }
  return out.str();
}

int main(int argc, char** argv) {
  const double seconds = argc == 2 ? std::stod(argv[1]) : 50.0;
  if (!(seconds > 0 && seconds <= 120)) return 3;
  rclcpp::init(0, nullptr);
  auto node = std::make_shared<rclcpp::Node>("x2_command_gid_readonly_audit");
  using Message = aimdk_msgs::msg::JointCommandArray;
  using Clock = std::chrono::steady_clock;
  std::array<std::string, 4> groups{"leg", "waist", "arm", "head"};
  std::map<std::string, std::map<std::string, uint64_t>> counts;
  std::map<std::string, std::map<std::string, std::string>> known;
  std::map<std::string, std::map<std::string, std::string>> previous;
  std::array<rclcpp::Subscription<Message>::SharedPtr, 4> subscriptions;
  for (size_t i = 0; i < groups.size(); ++i) {
    auto group = groups[i];
    subscriptions[i] = node->create_subscription<Message>(
        "/aima/hal/joint/" + group + "/command",
        rclcpp::SensorDataQoS().keep_last(10),
        [&, group](Message::ConstSharedPtr, const rclcpp::MessageInfo& info) {
          ++counts[group][Hex(info.get_rmw_message_info().publisher_gid.data)];
        });
  }
  std::cout << "event,elapsed_s,group,gid,node,count\n";
  const auto start = Clock::now();
  auto next = start;
  while (rclcpp::ok() && std::chrono::duration<double>(Clock::now()-start).count() < seconds) {
    rclcpp::spin_some(node);
    auto now = Clock::now();
    if (now >= next) {
      const double elapsed = std::chrono::duration<double>(now-start).count();
      for (const auto& group : groups) {
        std::map<std::string, std::string> current;
        for (const auto& endpoint : node->get_publishers_info_by_topic(
                 "/aima/hal/joint/" + group + "/command")) {
          const auto gid = Hex(endpoint.endpoint_gid());
          current[gid] = endpoint.node_name();
          known[group][gid] = endpoint.node_name();
        }
        if (current != previous[group]) {
          for (const auto& pair : current)
            std::cout << "endpoint," << elapsed << ',' << group << ','
                      << pair.first << ',' << pair.second << ",0\n";
          previous[group] = current;
        }
        for (const auto& pair : counts[group])
          std::cout << "sample," << elapsed << ',' << group << ','
                    << pair.first << ',' << known[group][pair.first]
                    << ',' << pair.second << '\n';
      }
      std::cout.flush();
      next = now + std::chrono::seconds(1);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  bool passed = rclcpp::ok();
  for (const auto& group : groups) {
    if (counts[group].empty()) passed = false;
    for (const auto& pair : counts[group]) {
      const auto name = known[group][pair.first];
      if (name != "mc_ros2_node") passed = false;
      std::cout << "final," << std::chrono::duration<double>(Clock::now()-start).count()
                << ',' << group << ',' << pair.first << ',' << name
                << ',' << pair.second << '\n';
    }
    for (const auto& pair : known[group])
      std::cout << "known,0," << group << ',' << pair.first << ','
                << pair.second << ',' << counts[group][pair.first] << '\n';
  }
  std::cout << "result,0,,,," << (passed ? "PASS" : "FAIL") << std::endl;
  rclcpp::shutdown();
  return passed ? 0 : 2;
}
