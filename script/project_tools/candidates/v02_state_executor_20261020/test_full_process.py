#!/usr/bin/env python3
"""Synthetic domain-only process integration. Never run in robot domain 0."""
import argparse, hashlib, json, os, re, signal, subprocess, time, uuid
from pathlib import Path
if os.environ.get('ROS_DOMAIN_ID') != '173' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
    raise SystemExit('Refusing test outside isolated domain173/localhost')
import rclpy
from rclpy.qos import qos_profile_sensor_data
from aimdk_msgs.msg import JointState, JointStateArray, JointCommandArray
from sensor_msgs.msg import Imu
ap=argparse.ArgumentParser()
ap.add_argument('--workspace',required=True)
ap.add_argument('--independent-writer',action='store_true')
ap.add_argument('--case',choices=['healthy','malformed','stdin','stale','late_malformed','fake_publish','fake_stale','fake_hold','fake_block','fake_partial0','fake_partial1','fake_partial2','fake_partial3'],required=True)
a=ap.parse_args(); ws=Path(a.workspace)
out=ws/('integration_'+a.case+'_'+str(time.time_ns()));out.mkdir()
params=(ws/'src/x2/agi_x2_deploy_onnx_ref/include/policy_parameters.hpp').read_text()
names=re.findall(r'"([a-z_]+joint)"',params.split('mujoco_joint_names = {')[1].split('};')[0])
angles=[float(x) for x in re.findall(r'^\s*([-0-9.]+),',params.split('default_angles = {')[1].split('};')[0],re.M)]
assert len(names)==len(angles)==31
prefix='/plan_a_fake_'+uuid.uuid4().hex
def topic(name):return prefix+name
# Only the session/process group created by our successful Popen is owned here.
def stop_child():
    for sig,wait_s in [(signal.SIGINT,2),(signal.SIGTERM,2),(signal.SIGKILL,2)]:
        try:os.killpg(child.pid,sig)
        except ProcessLookupError:
            child.wait(timeout=2)  # Reap even if the group disappeared first.
            return
        deadline=time.monotonic()+wait_s
        while time.monotonic()<deadline:
            child.poll()  # Reap the leader, but do not abandon surviving group members.
            try:os.killpg(child.pid,0)
            except ProcessLookupError:
                child.wait(timeout=2)
                return
            time.sleep(0.02)
    # SIGKILL was sent to the entire owned group; reap our direct child boundedly.
    child.wait(timeout=2)

node=None;log=None;child=None
state_publisher_matches={g:0 for g in ['leg','waist','arm','head','imu']}
command_sink_matches={g:0 for g in ['leg','waist','arm','head']}
try:
    rclpy.init()
    node=rclpy.create_node('plan_a_full_process_test')
    cmd_counts={g:0 for g in ['leg','waist','arm','head']}
    pubs=[];msgs=[];subs=[]
    damping_counts={g:0 for g in cmd_counts}
    last_commands={}
    command_times={g:[] for g in cmd_counts}
    block_counts_before=None;block_counts_after=None
    expected_damping={'leg':[9,8,8,4,7.5,3]*2,'waist':[5,6.25,4],'arm':[4,5,3,3,3,1,1]*2,'head':[0,0]}
    for g,start,n in [('leg',0,12),('waist',12,3),('arm',15,14),('head',29,2)]:
        p=node.create_publisher(JointStateArray,topic('/aima/hal/joint/'+g+'/state'),qos_profile_sensor_data)
        m=JointStateArray()
        for i in range(start,start+n):
            j=JointState();j.name=names[i];j.position=angles[i];m.joints.append(j)
        if a.case=='malformed' and g=='leg':m.joints=[]
        pubs.append(p);msgs.append(m)
        def receive(msg,group=g):
            cmd_counts[group]+=1
            command_times[group].append(time.monotonic())
            last_commands[group]=msg
            # Exact native vendor damping vector, including undamped head.
            if len(msg.joints)==len(expected_damping[group]) and all(j.stiffness==0.0 and abs(j.damping-kd)<1e-6 for j,kd in zip(msg.joints,expected_damping[group])):
                damping_counts[group]+=1
        subs.append(node.create_subscription(JointCommandArray,topic('/aima/hal/joint/'+g+'/command'),receive,qos_profile_sensor_data))
    imu_pub=node.create_publisher(Imu,topic('/aima/hal/imu/torso/state'),qos_profile_sensor_data)
    imu=Imu();imu.orientation.w=1.0
    state_publishers=dict(zip(cmd_counts,pubs));state_publishers['imu']=imu_pub
    command_sinks=dict(zip(cmd_counts,subs))
    def track_matches():
        for group,publisher in state_publishers.items():
            state_publisher_matches[group]=max(state_publisher_matches[group],publisher.get_subscription_count())
        for group,sink in command_sinks.items():
            command_sink_matches[group]=max(command_sink_matches[group],node.count_publishers(topic('/aima/hal/joint/'+group+'/command')))
    argv=[str(ws/'native-build/x2_deploy_onnx_ref'),'--model','/agibot/data/home/agi/gear-sonic/policies/x2_sonic_s1ft16000_g1.onnx','--no-publish','--max-duration','3','--return-seconds','0','--hold-for-mc-timeout-s','0','--mc-mode-poll-s','0','--intra-op-threads','3','--log-dir',str(out/'csv')]
    if a.case!='stdin':argv+=['--autostart-after','0']
    if a.independent_writer:argv+=['--independent-writer-offline','--offline-fake-prefix',prefix]
    if a.case=='fake_block':argv+=['--offline-control-block-ms','300']
    if a.case.startswith('fake_partial'):argv+=['--offline-publish-fail-group',a.case[-1]]
    argv+=['--ros-args']
    for g in ['leg','waist','arm','head']:
        for kind in ['state','command']:
            name='/aima/hal/joint/'+g+'/'+kind
            argv+=['-r',name+':='+topic(name)]
    name='/aima/hal/imu/torso/state';argv+=['-r',name+':='+topic(name)]
    if a.case in ['fake_publish','fake_stale','fake_hold','fake_block'] or a.case.startswith('fake_partial'):
        # All real joint command endpoints MUST have explicit unique remaps before
        # removing no-publish for this positive-control fake sink test.
        for g in ['leg','waist','arm','head']:
            name='/aima/hal/joint/'+g+'/command'
            if name+':='+topic(name) not in argv:raise SystemExit('Missing command remap')
        argv.remove('--no-publish')
    if a.case=='fake_hold':
        argv[argv.index('--return-seconds')+1]='0.4'
        argv[argv.index('--hold-for-mc-timeout-s')+1]='5'
        i=argv.index('--ros-args')
        argv[i:i]=['--hold-for-mc-sentinel',str(out/'hold.ready'),'--hold-for-mc-exit-sentinel',str(out/'hold.exit')]
    env=os.environ.copy();env['ROS_DOMAIN_ID']='173';env['ROS_LOCALHOST_ONLY']='1';env['X2_PLANT']='vendor_20260823';env['X2_REPO_ROOT']='/agibot/data/home/agi/gear-sonic';env['X2_ANCHOR_ORI_MODE']='body'
    identity={'binary_sha256':hashlib.sha256(Path(argv[0]).read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(Path(argv[argv.index('--model')+1]).read_bytes()).hexdigest(),'parameters_sha256':hashlib.sha256(params.encode()).hexdigest(),'argv':argv,'domain':env['ROS_DOMAIN_ID'],'localhost':env['ROS_LOCALHOST_ONLY'],'rmw':env.get('RMW_IMPLEMENTATION','default')}
    (out/'identity.json').write_text(json.dumps(identity,indent=2))
    log=(out/'runtime.log').open('w')
    child=subprocess.Popen(argv,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    start=time.monotonic();control_at=None;signalled=False;timed_out=False
    while child.poll() is None and time.monotonic()-start<18:
        track_matches()
        text=(out/'runtime.log').read_text()
        if '-> CONTROL' in text and control_at is None:control_at=time.monotonic()
        if 'OFFLINE_CONTROL_BLOCK_BEGIN' in text and block_counts_before is None:block_counts_before=cmd_counts.copy()
        if 'OFFLINE_CONTROL_BLOCK_END' in text and block_counts_after is None:block_counts_after=cmd_counts.copy()
        if a.case=='late_malformed' and control_at is not None and time.monotonic()-control_at>1:
            msgs[0].joints=[]
        send=not(a.case in ['stale','fake_stale'] and control_at is not None and time.monotonic()-control_at>1)
        if send:
            for p,m in zip(pubs,msgs):p.publish(m)
            imu_pub.publish(imu)
        if a.case=='stdin' and 'WAIT_FOR_CONTROL' in text and not signalled:
            child.send_signal(signal.SIGINT);signalled=True
        if a.case=='fake_hold' and (out/'hold.ready').exists():
            (out/'hold.exit').touch()
        if a.case in ['stale','late_malformed','fake_stale'] and 'E-STOP -> PURE DAMPING' in text and not signalled and (a.case!='fake_stale' or all(v>2 for v in damping_counts.values())):
            child.send_signal(signal.SIGINT);signalled=True
        rclpy.spin_once(node,timeout_sec=0.002)
    if child.poll() is None:
        timed_out=True  # The single outer cleanup owns escalation/reaping.
finally:
    # Each cleanup gets a chance even if an earlier close/stop/drain fails.
    try:
        if child is not None:stop_child()
    finally:
        try:
            if child is not None and child.stdin is not None:child.stdin.close()
        finally:
            try:
                if log is not None:log.close()
            finally:
                try:
                    if child is not None and node is not None:
                        drain_until=time.monotonic()+0.2
                        while time.monotonic()<drain_until:
                            track_matches()
                            rclpy.spin_once(node,timeout_sec=0.002)
                finally:
                    try:
                        if node is not None:node.destroy_node()
                    finally:
                        if rclpy.ok():rclpy.shutdown()
text=(out/'runtime.log').read_text()
checks={'state_publishers_matched':all(v>0 for v in state_publisher_matches.values()),'command_sinks_matched':all(v>0 for v in command_sink_matches.values()),'independent_banner':(not a.independent_writer or 'INDEPENDENT_WRITER_OFFLINE' in text),'no_commands':sum(cmd_counts.values())==0,'bounded_exit':not timed_out,'isolation_banner':'STATE_EXECUTOR_ISOLATED' in text}
if a.case in ['fake_publish','fake_stale','fake_hold','fake_block'] or a.case.startswith('fake_partial'):
    del checks['no_commands']  # positive-control publishes ONLY to remapped fake topics
    checks.update(fake_commands_observed=all(v>50 for v in cmd_counts.values()),release_ack=('WRITER_RELEASE_ACK' in text),exit_ok=child.returncode==0,control='-> CONTROL' in text)
if a.case=='fake_block':checks.update(block_seen=block_counts_before is not None and block_counts_after is not None,writer_progress=block_counts_before is not None and block_counts_after is not None and all(block_counts_after[g]-block_counts_before[g]>20 for g in cmd_counts),no_native_stale='E-STOP -> PURE DAMPING' not in text)
if a.case.startswith('fake_partial'):
    checks.pop('release_ack');checks['exit_ok']=child.returncode==5
    checks['fake_commands_observed']=all(v>=40 for v in cmd_counts.values())
    checks.update(injected='OFFLINE_PARTIAL_INJECTION' in text,fault_channel='OFFLINE_WRITER_FAILURE' in text,no_false_ack='WRITER_RELEASE_ACK' not in text,no_hold_ack='HOLD_FIRST_BATCH_ACK' not in text)
if a.case=='fake_stale':
    del checks['release_ack']  # forced isolated signal shutdown is NOT graceful release
    checks.update(damping_on_all_groups=all(v>2 for v in damping_counts.values()),native_guard='E-STOP -> PURE DAMPING' in text,signal_sent=signalled)
if a.case=='fake_hold':checks.update(hold_ready_seen=(out/'hold.exit').exists(),ready_cleared=not (out/'hold.ready').exists(),first_batch_ack='HOLD_FIRST_BATCH_ACK' in text)
if a.case=='healthy':checks.update(control='-> CONTROL' in text,no_stale='E-STOP -> PURE DAMPING' not in text,exit_ok=child.returncode==0)
if a.case=='malformed':checks.update(failure='State receiver failed before CONTROL' in text,exit_four=child.returncode==4,no_control='-> CONTROL' not in text)
if a.case=='stdin':checks.update(signal_sent=signalled,exit_ok=child.returncode==0,no_control='-> CONTROL' not in text)
if a.case in ['stale','late_malformed']:checks.update(exit_ok=child.returncode in [0,4],signal_sent=signalled,native_guard='E-STOP -> PURE DAMPING' in text,control='-> CONTROL' in text)
intervals={g:[1000*(b-a) for a,b in zip(ts,ts[1:])] for g,ts in command_times.items()}
interval_summary={g:dict(count=len(xs),mean_ms=sum(xs)/len(xs) if xs else None,max_ms=max(xs) if xs else None) for g,xs in intervals.items()}
result={'block_counts_before':block_counts_before,'block_counts_after':block_counts_after,'state_publisher_matches':state_publisher_matches,'command_sink_matches':command_sink_matches,'independent_writer':a.independent_writer,'receive_intervals':interval_summary,'case':a.case,'exit':child.returncode,'commands':cmd_counts,'damping_commands':damping_counts,'checks':checks,'output':str(out)}
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
raise SystemExit(0 if all(checks.values()) else 1)
