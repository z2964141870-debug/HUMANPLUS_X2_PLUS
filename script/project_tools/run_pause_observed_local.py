#!/usr/bin/env python3
"""One explicitly authorized no-custom MC diagnostic; operator runs manually."""
import json,subprocess,time,pathlib,sys
from observation_window_gate import check_window
out=pathlib.Path('/tmp/x2-pause-covered-'+str(time.time_ns()));out.mkdir();print('OUTPUT='+str(out),flush=True)
s0=['ssh','-o','BatchMode=yes','-S','/tmp/x2-round1-20261020/soc0-restored.sock','-p','22040','run@127.0.0.1']
s1=['ssh','-o','BatchMode=yes','-S','/tmp/x2-round1-20261020/soc1-restored.sock','-p','22041','agi@127.0.0.1']
if sys.argv[1:]!=['--execute-authorized-mc-pause']:raise SystemExit('explicit execute flag required')
# Pin exact target from known MC comm, rejecting zero/multiple candidates.
identity_code="import sys,json,subprocess;sys.path.insert(0,'/tmp');import x2_mc_pause1310 as m;p=subprocess.check_output(['pgrep','-x','mc_main'],text=True).split();assert len(p)==1;print(json.dumps(m.snapshot(int(p[0]))))"
r=subprocess.run(s0+['/usr/bin/python3 -c '+__import__('shlex').quote(identity_code)],capture_output=True,text=True,timeout=10);r.check_returncode();identity=json.loads(r.stdout)
# Reject active custom controller before any action; bracketed regex avoids matching query shell.
for host in [s0,s1]:
 r=subprocess.run(host+["pgrep -af '[x]2_deploy_onnx_ref'"],capture_output=True,text=True,timeout=10)
 if r.returncode!=1:raise RuntimeError('custom controller found or preflight failed: '+r.stdout+r.stderr)
script='source /opt/ros/humble/setup.bash; export PYTHONPATH=/agibot/data/home/agi/aimdk_msgs_pc2/local/lib/python3.10/dist-packages:\$PYTHONPATH; export LD_LIBRARY_PATH=/agibot/data/home/agi/aimdk_msgs_pc2/lib:\$LD_LIBRARY_PATH; export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0; /usr/bin/python3 /agibot/data/home/agi/x2-v02-state-executor-20261021.rMYM0G/x2_hal_readonly_probe.py --seconds 45 --tag soc1'
# shell quote keeps ROS environment expansions on the source side.
script=script.replace('\$','$')
start=time.monotonic();observer=subprocess.Popen(s1+['bash -c '+__import__('shlex').quote(script)],stdout=subprocess.PIPE,stderr=(out/'observer.stderr').open('w'),text=True)
ready=False;paused=False;owners={};pause_result=None
try:
 with (out/'observer.jsonl').open('w') as log:
  for line in observer.stdout:
   log.write(line);log.flush()
   try:x=json.loads(line)
   except ValueError:continue
   if x.get('event')=='endpoints' and x.get('tag','').startswith('cmd_'):owners[x['tag']]=[p['node'] for p in x['publishers']]
   if x.get('event')=='baseline_ready':ready=True
   if not paused and ready and x.get('event')=='sample':
    streams=x['streams'];required=['state_'+g for g in ['leg','waist','arm','head']]+['imu_torso']+['cmd_'+g for g in ['leg','waist','arm','head']]
    fresh=all(streams[k]['age_s'] is not None and 0<=streams[k]['age_s']<.5 for k in required)
    if not fresh or not all(owners.get('cmd_'+g)==['mc_ros2_node'] for g in ['leg','waist','arm','head']):raise RuntimeError('not fresh sole official')
    if not check_window(observer_running=observer.poll() is None,ready=ready,observer_started_at=start,now=time.monotonic(),duration=45.):raise RuntimeError('observation budget exhausted')
    cmd=['/usr/bin/python3','/tmp/x2_mc_pause1310.py','--pid',str(identity['pid']),'--boot-id',identity['boot_id'],'--start-ticks',str(identity['start_ticks']),'--argv-sha256',identity['argv_sha256'],'--execute','SUSPENDED_MC_PAUSE_2S','--pause-seconds','2']
    paused=True;r=subprocess.run(s0+[__import__('shlex').join(cmd)],capture_output=True,text=True,timeout=10)
    (out/'pause.jsonl').write_text(r.stdout);(out/'pause.stderr').write_text(r.stderr);r.check_returncode();pause_result=r.stdout;print(r.stdout,flush=True)
  rc=observer.wait(timeout=5)
 if rc or not paused:raise RuntimeError('observer failed or pause not executed')
 print('CAPTURE_COMPLETE='+str(out),flush=True)
finally:
 if observer.poll() is None:
  observer.terminate()
  try:observer.wait(timeout=3)
  except subprocess.TimeoutExpired:observer.kill();observer.wait(timeout=3)
