import unittest
from observation_window_gate import check_window
class Gate(unittest.TestCase):
 def check(self,**kwargs):
  args=dict(observer_running=True,ready=True,observer_started_at=100.,now=105.,duration=60.);args.update(kwargs);return check_window(**args)
 def test_ready_and_sufficient(self):self.assertTrue(self.check())
 def test_old_fifteen_second_window_rejected(self):self.assertFalse(self.check(duration=15.))
 def test_observer_finished(self):self.assertFalse(self.check(observer_running=False))
 def test_no_discovery(self):self.assertFalse(self.check(ready=False))
 def test_late_action_rejected(self):self.assertFalse(self.check(now=150.))
 def test_backward_clock(self):self.assertFalse(self.check(now=99.))
 def test_nan(self):self.assertFalse(self.check(now=float('nan')))
 def test_infinite_duration(self):self.assertFalse(self.check(duration=float('inf')))
 def test_negative_pause(self):self.assertFalse(self.check(pause_s=-2.))
 def test_exact_budget(self):self.assertTrue(self.check(now=143.))
 def test_below_budget(self):self.assertFalse(self.check(now=143.01))
if __name__=='__main__':unittest.main()
