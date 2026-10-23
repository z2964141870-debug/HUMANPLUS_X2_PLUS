"""Offline handoff order specification; no OS signals or robot APIs."""
import itertools,unittest

def allowed_cont(*,identity_valid,custom_started,release_ack,custom_exited,pending_generation):
    # ACK is local publication-gate drain, not DDS delivery/queue drain proof.
    return identity_valid and not pending_generation and (not custom_started or (release_ack and custom_exited))

class Contract(unittest.TestCase):
 def test_all_combinations(self):
  for identity,started,ack,exited,pending in itertools.product([False,True],repeat=5):
   result=allowed_cont(identity_valid=identity,custom_started=started,release_ack=ack,custom_exited=exited,pending_generation=pending)
   if result:
    self.assertTrue(identity);self.assertFalse(pending)
    self.assertTrue(not started or (ack and exited))
 def test_timeout_is_not_release(self):
  self.assertFalse(allowed_cont(identity_valid=True,custom_started=True,release_ack=False,custom_exited=False,pending_generation=False))
 def test_crash_is_not_release(self):
  self.assertFalse(allowed_cont(identity_valid=True,custom_started=True,release_ack=False,custom_exited=True,pending_generation=False))
 def test_ack_only_is_not_exit(self):
  self.assertFalse(allowed_cont(identity_valid=True,custom_started=True,release_ack=True,custom_exited=False,pending_generation=False))
 def test_pending_inference_blocks_rearm(self):
  self.assertFalse(allowed_cont(identity_valid=True,custom_started=True,release_ack=True,custom_exited=True,pending_generation=True))
 def test_no_custom_short_pause(self):
  self.assertTrue(allowed_cont(identity_valid=True,custom_started=False,release_ack=False,custom_exited=False,pending_generation=False))
 def test_completed_custom(self):
  self.assertTrue(allowed_cont(identity_valid=True,custom_started=True,release_ack=True,custom_exited=True,pending_generation=False))
if __name__=='__main__':unittest.main()
