#!/usr/bin/env python3
"""Isolated synthetic DDS lifecycle diagnostic, never uses HAL topics."""
import argparse,json,os,subprocess,sys,time,uuid
p=argparse.ArgumentParser();p.add_argument('--reader');a=p.parse_args()
assert os.environ.get('ROS_DOMAIN_ID')=='173' and os.environ.get('ROS_LOCALHOST_ONLY')=='1'
import rclpy
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from sensor_msgs.msg import Imu
rclpy.init(args=[]);node=rclpy.create_node('synthetic_lifecycle_'+str(os.getpid()))
q=QoSProfile(depth=5,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE if a.reader else DurabilityPolicy.TRANSIENT_LOCAL)
child=None
try:
 if a.reader:
  assert a.reader.startswith('/plan_a_lifecycle_')
  count=[0];sub=node.create_subscription(Imu,a.reader,lambda m:count.__setitem__(0,count[0]+1),q)
  end=time.monotonic()+1.2
  while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.002)
  print(json.dumps({'received':count[0]}));raise SystemExit(0 if count[0]>10 else 1)
 topic='/plan_a_lifecycle_'+uuid.uuid4().hex
 pub=node.create_publisher(Imu,topic,q);msg=Imu();msg.orientation.w=1.
 costs=[];intervals=[];codes=[];children=[];last=None
 for cycle in range(8):
  child=subprocess.Popen([sys.executable,__file__,'--reader',topic],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
  end=time.monotonic()+1.8
  while time.monotonic()<end:
   now=time.monotonic()
   if last is not None:intervals.append(now-last)
   last=now;t=time.monotonic();pub.publish(msg);costs.append(time.monotonic()-t)
   rclpy.spin_once(node,timeout_sec=0);time.sleep(.002)
  output,_=child.communicate(timeout=3);codes.append(child.returncode)
  try:children.append(json.loads(output.strip().splitlines()[-1]))
  except (ValueError,IndexError):children.append({'unparsed':output[-300:]})
  child=None
 result={'topic':topic,'cycles':len(codes),'reader_exit_codes':codes,'readers':children,'publish_calls':len(costs),'max_publish_ms':1000*max(costs),'mean_publish_ms':1000*sum(costs)/len(costs),'max_loop_interval_ms':1000*max(intervals),'scope':'same-host Python rmw surrogate; not vendor C++ HAL, not MC handoff, no real commands'}
 print(json.dumps(result),flush=True)
 assert all(c==0 for c in codes)
finally:
 if child is not None and child.poll() is None:
  child.terminate()
  try:child.wait(timeout=3)
  except subprocess.TimeoutExpired:child.kill();child.wait(timeout=3)
 node.destroy_node();rclpy.shutdown()
