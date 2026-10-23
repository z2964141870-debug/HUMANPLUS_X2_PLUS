#include "command_publication_gate.hpp"
#include <array>
#include <atomic>
#include <cassert>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace std::chrono_literals;
int main(){
  using Gate=agi_x2::CommandPublicationGate<std::array<int,4>>;
  Gate reversible;
  assert(reversible.Commit({7,7,7,7},1,false));
  assert(!reversible.Publish([](const auto&){assert(false);}));
  assert(reversible.Commit({8,8,8,8},2,true));
  assert(reversible.Publish([](const auto& e){assert(e.mode==2 && e.enabled && e.command[0]==8);}));
  assert(reversible.PublishedGeneration()==2);
  assert(reversible.Commit({9,9,9,9},3,false));
  assert(!reversible.Publish([](const auto&){assert(false);}));
  reversible.Release();assert(!reversible.Commit({8,8,8,8},2,true));
  // Disable must drain an active complete batch, but unlike Release can rearm.
  Gate disabling;assert(disabling.Commit({3,3,3,3},1));
  std::promise<void> batch_entered,batch_resume;
  auto resume=batch_resume.get_future();
  std::thread active([&]{assert(disabling.Publish([&](const auto& e){
    batch_entered.set_value();resume.wait();
    for(int v:e.command)assert(v==3);
  }));});
  batch_entered.get_future().wait();
  auto disabled=std::async(std::launch::async,[&]{return disabling.Commit({0,0,0,0},0,false);});
  assert(disabled.wait_for(10ms)==std::future_status::timeout);
  batch_resume.set_value();active.join();assert(disabled.get());
  assert(!disabling.Publish([](const auto&){assert(false);}));
  assert(disabling.Commit({4,4,4,4},2,true));
  assert(disabling.Publish([](const auto& e){for(int v:e.command)assert(v==4);}));
  // Throw at each partial batch boundary; never mark that generation complete.
  for(int fail_group=0;fail_group<4;++fail_group){
    Gate partial;assert(partial.Commit({5,5,5,5},1));
    int sent=0;bool failed=false;
    try{partial.Publish([&](const auto& e){for(int i=0;i<4;++i){
      if(i==fail_group)throw std::runtime_error("group failure");
      assert(e.command[i]==5);++sent;
    }});}catch(const std::runtime_error&){failed=true;}
    assert(failed && sent==fail_group && partial.PublishedGeneration()==0);
    assert(!partial.Release());assert(!partial.Commit({6,6,6,6},1));
  }
  // Race release against a failing batch BEFORE callback outer catch runs.
  Gate racing;assert(racing.Commit({1,1,1,1}));
  std::promise<void> racing_entered,racing_continue;
  auto continue_batch=racing_continue.get_future();
  std::atomic<bool> outer_catch{false};
  std::thread failing([&]{try{racing.Publish([&](const auto&){
    racing_entered.set_value();continue_batch.wait();
    throw std::runtime_error("partial publication");
  });}catch(...){outer_catch=true;}});
  racing_entered.get_future().wait();
  auto racing_release=std::async(std::launch::async,[&]{return racing.Release();});
  racing_continue.set_value();failing.join();
  assert(!racing_release.get());assert(outer_catch);
  assert(!racing.Commit({2,2,2,2}));
  Gate gate; int calls=0;
  assert(!gate.Publish([&](const auto&){++calls;}));
  assert(gate.Commit({1,1,1,1}));
  std::promise<void> inside,allow_finish;
  auto allowed=allow_finish.get_future();
  std::atomic<int> groups{0};
  std::thread publisher([&]{assert(gate.Publish([&](const auto& envelope){
    assert(envelope.generation==1);
    ++groups;inside.set_value();allowed.wait();
    for(int i=1;i<4;++i){assert(envelope.command[i]==1);++groups;}
  }));});
  inside.get_future().wait();
  auto release=std::async(std::launch::async,[&]{gate.Release();});
  assert(release.wait_for(10ms)==std::future_status::timeout);
  allow_finish.set_value();publisher.join();release.get();
  assert(groups==4);
  assert(!gate.Commit({2,2,2,2}));
  for(int i=0;i<1000;++i)assert(!gate.Publish([&](const auto&){++calls;}));
  gate.Release();assert(calls==0);
  Gate throwing;assert(throwing.Commit({1,1,1,1}));
  bool caught=false;
  try{throwing.Publish([](const auto&){throw std::runtime_error("partial batch");});}
  catch(const std::runtime_error&){caught=true;}
  assert(caught);assert(throwing.PublishedGeneration()==0);assert(!throwing.Release());
  assert(!throwing.Publish([](const auto&){}));
  std::cout<<"PASS: release drains entire in-flight batch; terminal no-rearm/no-late-publish; exception unlock\n";
}
