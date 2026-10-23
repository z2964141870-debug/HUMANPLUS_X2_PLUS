import unittest
from mc_recovery_evidence import recovered,GROUPS
class Recovery(unittest.TestCase):
 def setUp(self):
  self.b={g:10 for g in GROUPS};self.a={g:20 for g in GROUPS};self.o={g:['mc_ros2_node'] for g in GROUPS};self.ages={k:.01 for k in ['command_'+g for g in GROUPS]+['state_'+g for g in GROUPS]+['imu']}
 def ok(self,**kwargs):
  x=dict(before_counts=self.b,after_counts=self.a,owners=self.o,ages=self.ages,now=3.,window_started=0.);x.update(kwargs);return recovered(**x)
 def test_valid_evidence(self):self.assertTrue(self.ok())
 def test_graph_without_flow(self):self.assertFalse(self.ok(after_counts=self.b))
 def test_missing_group(self):self.o.pop('head');self.assertFalse(self.ok())
 def test_dual_writer(self):self.o['leg'].append('x2_deploy_onnx_ref');self.assertFalse(self.ok())
 def test_wrong_owner(self):self.o['arm']=['x2_deploy_onnx_ref'];self.assertFalse(self.ok())
 def test_stale_command(self):self.ages['command_head']=1.;self.assertFalse(self.ok())
 def test_stale_imu(self):self.ages['imu']=1.;self.assertFalse(self.ok())
 def test_missing_age(self):self.ages.pop('state_waist');self.assertFalse(self.ok())
 def test_negative_clock(self):self.ages['imu']=-.1;self.assertFalse(self.ok())
 def test_short_window(self):self.assertFalse(self.ok(now=.5))
 def test_nan_clock(self):self.ages['imu']=float('nan');self.assertFalse(self.ok())
if __name__=='__main__':unittest.main()
