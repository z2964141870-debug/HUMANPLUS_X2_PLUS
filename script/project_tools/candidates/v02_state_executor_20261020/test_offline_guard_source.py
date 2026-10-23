#!/usr/bin/env python3
"""Compile the actual guard body against a fake resolver: no ROS/network.
This tests C++ guard decisions only, NOT ROS remap API or full binary build.
"""
from pathlib import Path
import subprocess,tempfile
root=Path(__file__).parent
source=(root/'x2_deploy_onnx_ref.cpp').read_text()
start=source.index('      const char* domain = std::getenv("ROS_DOMAIN_ID");',source.index('// Fail CLOSED'))
end=source.index('      RCLCPP_WARN(node->get_logger(),',start)
guard=source[start:end]
cpp=r'''
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
struct Cli {std::string offline_fake_prefix="/plan_a_fake_unit", input_type="motion_file",motion_path,token_model_path,dual_model_path,imu_topic; double mc_mode_poll_s=0;bool whole_body_teleop=false;int smpl_intent_port=0,zmq_debug_port=0;};
struct Node {std::string escape; Node* get_node_topics_interface(){return this;} std::string resolve_topic_name(const std::string& s){return s==escape?s:"/plan_a_fake_unit"+s;}};
void check(const Cli& cli,const std::shared_ptr<Node>& node){
''' + guard + r'''
}
int main(){
setenv("ROS_DOMAIN_ID","173",1);setenv("ROS_LOCALHOST_ONLY","1",1);
Cli cli;auto node=std::make_shared<Node>();int passed=0;
auto refused=[&]{bool rejected=false;try{check(cli,node);}catch(const std::runtime_error&){rejected=true;}if(!rejected)throw std::runtime_error("Guard incorrectly allowed case");++passed;};
check(cli,node);++passed;
setenv("ROS_DOMAIN_ID","0",1);refused();setenv("ROS_DOMAIN_ID","173",1);
unsetenv("ROS_DOMAIN_ID");refused();setenv("ROS_DOMAIN_ID","173",1);
setenv("ROS_LOCALHOST_ONLY","0",1);refused();setenv("ROS_LOCALHOST_ONLY","1",1);
for(const auto& s:{"", "/aima/hal", "/plan_a_fake_", "/plan_a_fake_x/y", "/plan_a_fake_x-y"}){cli.offline_fake_prefix=s;refused();}cli.offline_fake_prefix="/plan_a_fake_unit";
for(const auto& g:{"leg","waist","arm","head"})for(const auto& k:{"state","command"}){node->escape=std::string("/aima/hal/joint/")+g+"/"+k;refused();}node->escape="/aima/hal/imu/torso/state";refused();node->escape="";
cli.imu_topic="/escape";refused();cli.imu_topic="";
cli.mc_mode_poll_s=1;refused();cli.mc_mode_poll_s=0;
cli.input_type="zmq";refused();cli.input_type="motion_file";
cli.motion_path="motion";refused();cli.motion_path="";
cli.whole_body_teleop=true;refused();cli.whole_body_teleop=false;
cli.token_model_path="token";refused();cli.token_model_path="";
cli.dual_model_path="dual";refused();cli.dual_model_path="";
cli.smpl_intent_port=1;refused();cli.smpl_intent_port=0;
cli.zmq_debug_port=1;refused();
std::cout<<"PASS "<<passed<<" actual-source guard decisions (mock resolver, NOT ROS API)\n";
}
'''
with tempfile.TemporaryDirectory(prefix='x2-guard-') as d:
    path=Path(d);(path/'guard.cpp').write_text(cpp)
    subprocess.run(['clang++','-std=c++17','-fsanitize=address,undefined',str(path/'guard.cpp'),'-o',str(path/'test')],check=True)
    subprocess.run([str(path/'test')],check=True)
