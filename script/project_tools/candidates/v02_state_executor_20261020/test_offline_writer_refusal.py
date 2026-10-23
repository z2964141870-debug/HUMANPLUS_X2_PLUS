#!/usr/bin/env python3
"""Negative launch tests. --no-publish is unconditional; never starts control."""
import argparse,json,os,subprocess,uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);a=p.parse_args()
ws=Path(a.workspace);prefix='/plan_a_fake_'+uuid.uuid4().hex
base=[str(ws/'native-build/x2_deploy_onnx_ref'),'--model','/agibot/data/home/agi/gear-sonic/policies/x2_sonic_s1ft16000_g1.onnx','--no-publish','--mc-mode-poll-s','0','--independent-writer-offline','--offline-fake-prefix',prefix]
remaps=[]
for g in ['leg','waist','arm','head']:
    for k in ['state','command']:
        n='/aima/hal/joint/'+g+'/'+k;remaps+=['-r',n+':='+prefix+n]
n='/aima/hal/imu/torso/state';remaps+=['-r',n+':='+prefix+n]
cases=[('wrong_domain',{'ROS_DOMAIN_ID':'172'},base,remaps),
       ('localhost_off',{'ROS_LOCALHOST_ONLY':'0'},base,remaps),
       ('missing_imu_remap',{},base,remaps[:-2]),
       ('missing_command_remap',{},base,remaps[4:]),
       ('imu_override_escape',{},base+['--imu-topic','/escape_imu'],remaps),
       ('mc_service_enabled',{},base+['--mc-mode-poll-s','1'],remaps),
       ('prefix_escape',{},base+['--offline-fake-prefix','/aima/hal'],remaps),
       ('global_remap_escape',{},base,['-r','/aima/hal/joint/head/command:=/escape']+remaps),
       ('zmq_enabled',{},base+['--input-type','zmq'],remaps),
       ('fault_without_offline_flag',{},[x for x in base if x!='--independent-writer-offline']+['--offline-control-block-ms','300'],remaps),
       ('partial_without_offline_flag',{},[x for x in base if x!='--independent-writer-offline']+['--offline-publish-fail-group','1'],remaps)]
results=[]
for name,updates,argv,rs in cases:
    env=os.environ.copy();env.update(ROS_DOMAIN_ID='173',ROS_LOCALHOST_ONLY='1');env.update(updates)
    r=subprocess.run(argv+['--ros-args']+rs,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=12)
    ok=r.returncode==2 and ('INDEPENDENT_WRITER_OFFLINE CLOSED:' in r.stdout or (name in ['fault_without_offline_flag','partial_without_offline_flag'] and '--independent-writer-offline' in r.stdout and 'Argument error:' in r.stdout)) and '-> CONTROL' not in r.stdout and 'STATE_EXECUTOR_ISOLATED' not in r.stdout
    results.append(dict(case=name,exit=r.returncode,passed=ok,tail=r.stdout[-1800:]))
print(json.dumps(results,indent=2));raise SystemExit(0 if all(x['passed'] for x in results) else 1)
