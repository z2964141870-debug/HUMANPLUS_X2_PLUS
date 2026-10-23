import unittest
from pause_observation_session import session
class Clock:
 def __init__(self):self.t=0.
 def now(self):return self.t
 def sleep(self,s):self.t+=max(s,.001)
class Observer:
 def __init__(self,c):self.c=c;self.finished=0;self.ready_at=1.;self.dies_at=100.;self.custom=False;self.owner=True;self.health=True;self.delay=0.
 def start(self,seconds=60):self.c.t+=self.delay
 def running(self):return self.c.t<self.dies_at
 def ready(self):return self.c.t>=self.ready_at
 def fresh(self):return self.health
 def sole_official(self):return self.owner
 def no_custom_writer(self):return not self.custom
 def finish(self):self.finished+=1;return {'saved':True}
class Target:
 def __init__(self,c):self.c=c;self.pauses=0;self.pin_fail=False;self.pause_fail=False;self.resumed=True;self.after=None
 def pin_identity(self):
  if self.pin_fail:raise RuntimeError('identity changed')
  return 'pinned'
 def pause_two_seconds(self,identity):
  assert identity=='pinned';self.pauses+=1;self.c.t+=2.
  if self.after:self.after()
  if self.pause_fail:raise RuntimeError('ssh disconnected')
  return {'stop_to_cont_s':2.,'resumed':self.resumed}
class Session(unittest.TestCase):
 def setUp(self):self.c=Clock();self.o=Observer(self.c);self.t=Target(self.c);self.events=[]
 def runit(self):return session(self.o,self.t,self.c.now,self.c.sleep,self.events.append)
 def reject(self):
  with self.assertRaises((RuntimeError,ValueError)):self.runit()
  self.assertEqual(self.o.finished,1)
 def test_success(self):self.runit();self.assertEqual(self.t.pauses,1);self.assertEqual(self.o.finished,1);self.assertGreaterEqual(self.c.t,5.)
 def test_no_ready(self):self.o.ready_at=100.;self.reject();self.assertEqual(self.t.pauses,0)
 def test_dead_observer(self):self.o.dies_at=.5;self.reject();self.assertEqual(self.t.pauses,0)
 def test_custom_writer(self):self.o.custom=True;self.reject();self.assertEqual(self.t.pauses,0)
 def test_wrong_owner(self):self.o.owner=False;self.reject();self.assertEqual(self.t.pauses,0)
 def test_delayed_launch_budget(self):self.o.delay=50.;self.reject();self.assertEqual(self.t.pauses,0)
 def test_identity_change(self):self.t.pin_fail=True;self.reject();self.assertEqual(self.t.pauses,0)
 def test_unknown_pause_status(self):self.t.pause_fail=True;self.reject();self.assertEqual(self.t.pauses,1)
 def test_failed_cont(self):self.t.resumed=False;self.reject();self.assertEqual(self.t.pauses,1)
 def test_observer_dies_during_pause(self):self.o.dies_at=2.;self.reject();self.assertEqual(self.t.pauses,1)
 def test_recovery_no_flow(self):self.t.after=lambda:setattr(self.o,'health',False);self.reject();self.assertEqual(self.t.pauses,1)
if __name__=='__main__':unittest.main()
