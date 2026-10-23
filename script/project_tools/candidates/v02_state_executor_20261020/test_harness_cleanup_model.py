#!/usr/bin/env python3
"""Deterministic tests of actual cleanup function with modeled OS/processes.
Does NOT send signals or prove Linux process-group runtime behavior.
"""
import ast,signal,subprocess
from pathlib import Path
from types import SimpleNamespace
source=Path(__file__).with_name('test_full_process.py').read_text()
function=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='stop_child')
code=compile(ast.Module(body=[function],type_ignores=[]),'actual_stop_child','exec')
class Model:
    def __init__(self,exit_signal=None,exists=True,denied=False):
        self.pid=12345;self.clock=0;self.exists=exists;self.denied=denied
        self.exit_signal=exit_signal;self.signals=[];self.waits=[];self.polls=0
    def killpg(self,pid,sig):
        assert pid==self.pid
        if self.denied:raise PermissionError('modeled policy denial')
        if not self.exists:raise ProcessLookupError()
        if sig:
            self.signals.append(sig)
            if sig==self.exit_signal:self.exists=False
    def poll(self):self.polls+=1;return None if self.exists else 0
    def wait(self,timeout):
        self.waits.append(timeout)
        if self.exists:raise subprocess.TimeoutExpired('modeled child',timeout)
        return 0
    def monotonic(self):return self.clock
    def sleep(self,duration):self.clock+=duration

def run(m):
    ns=dict(child=m,os=SimpleNamespace(killpg=m.killpg),signal=signal,time=m)
    exec(code,ns);ns['stop_child']()
for exit_signal,expected in [(signal.SIGINT,[signal.SIGINT]),(signal.SIGTERM,[signal.SIGINT,signal.SIGTERM]),(signal.SIGKILL,[signal.SIGINT,signal.SIGTERM,signal.SIGKILL])]:
    m=Model(exit_signal);run(m);assert m.signals==expected and m.waits and m.clock<6.1
m=Model(exists=False);run(m);assert m.waits==[2] and not m.signals
m=Model(denied=True)
try:run(m)
except PermissionError:pass
else:raise AssertionError('must propagate policy denial')
m=Model()
try:run(m)
except subprocess.TimeoutExpired:pass
else:raise AssertionError('unreapable process must fail boundedly')
assert m.clock<6.1 and m.signals==[signal.SIGINT,signal.SIGTERM,signal.SIGKILL]
print('PASS 6 actual cleanup model cases: staged signals, vanished group, denial propagation, bounded timeout; NOT runtime signal proof')
