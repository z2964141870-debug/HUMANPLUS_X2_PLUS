cmake_minimum_required(VERSION 3.16)
project(x2_command_audit LANGUAGES CXX)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(aimdk_msgs REQUIRED)
set(AUDIT_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../../../x2_command_gid_audit.cpp"
    CACHE FILEPATH "Read-only command auditor source")
add_executable(x2_command_gid_audit "${AUDIT_SOURCE}")
ament_target_dependencies(x2_command_gid_audit rclcpp aimdk_msgs)
