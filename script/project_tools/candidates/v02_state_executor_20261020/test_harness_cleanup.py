#!/usr/bin/env python3
"""Endpoint-free tests of actual stop_child function extracted from harness."""
import ast,os,signal,subprocess,sys,time
from pathlib import Path
source=Path(__file__).with_name('test_full_process.py').read_text()
tree=ast.parse(source)
function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='stop_child')
module=ast.Module(body=[function],type_ignores=[])
ns=dict(os=os,signal=signal,time=time);exec(compile(module,'actual_harness_stop_child','exec'),ns)
for script in ['import time;time.sleep(60)', 'pass', 'import signal,time;signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(60)']:
    child=subprocess.Popen([sys.executable,'-c',script],start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
    try:
        if 'ready' in script:assert child.stdout.readline().strip()=='ready'
        if script=='pass':child.wait(timeout=2)
        ns['child']=child;start=time.monotonic();ns['stop_child']()
        assert child.poll() is not None
        assert time.monotonic()-start<8
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=2)
        child.stdout.close()
print('PASS actual harness cleanup: active child, exited child, signal-ignoring child; bounded owned group/reap')
