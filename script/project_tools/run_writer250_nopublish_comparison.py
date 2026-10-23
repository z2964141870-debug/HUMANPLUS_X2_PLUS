#!/usr/bin/env python3
"""Finite no-command comparison. Does not call EM, signal MC or change config."""
import json,os,pathlib,subprocess,time
w=pathlib.Path('/agibot/data/home/agi/x2-v02-state-executor-20261021.rMYM0G')
out=pathlib.Path(__import__('tempfile').mkdtemp(prefix='x2-serial250-nopublish.',dir='/agibot/data/home/agi'))
args=json.loads((w/'baseline_argv.json').read_text())
assert '--no-publish' in args and '--start-trigger-sentinel' not in args
print('OUTPUT='+str(out),flush=True)
(out/'argv.json').write_text(json.dumps(args,indent=2))
observer=None;child=None
try:
 with (out/'observer.jsonl').open('w') as obs,(out/'observer.stderr').open('w') as err,(out/'runtime.log').open('w') as log:
  observer=subprocess.Popen(['/usr/bin/python3',str(w/'x2_hal_readonly_probe.py'),'--seconds','30','--tag','soc1'],stdout=obs,stderr=err)
  time.sleep(3)
  if observer.poll() is not None:raise RuntimeError('observer exited before deploy')
  args+=['--log-dir',str(out/'csv')]
  child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
  try:rc=child.wait(timeout=25)
  except subprocess.TimeoutExpired:
   child.terminate();rc=child.wait(timeout=5)
   raise RuntimeError('no-publish deploy exceeded wall bound')
  orc=observer.wait(timeout=15)
  result={'deploy_exit':rc,'observer_exit':orc,'command_sending':False,'mc_changed':False}
  (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
  if rc or orc:raise SystemExit(1)
finally:
 for p in (child,observer):
  if p is not None and p.poll() is None:
   p.terminate()
   try:p.wait(timeout=3)
   except subprocess.TimeoutExpired:p.kill();p.wait(timeout=3)
