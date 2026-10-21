import signal
import unittest

from x2_mc_bounded_pause_probe import bounded_pause


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeProcess:
    def __init__(self, state="S", acknowledge=True, resume=True):
        self.current = state
        self.signals = []
        self.acknowledge = acknowledge
        self.resume = resume

    def state(self):
        return self.current

    def send(self, sig):
        self.signals.append(sig)
        if sig == signal.SIGSTOP and self.acknowledge:
            self.current = "T"
        if sig == signal.SIGCONT and self.resume:
            self.current = "S"


class BoundedPauseTests(unittest.TestCase):
    def run_pause(self, target, **kwargs):
        clock = FakeClock()
        bounded_pause(target, kwargs.pop("seconds", 2.0), clock=clock.clock,
                      sleep=clock.sleep, log=kwargs.pop("log", lambda *a, **kw: None),
                      **kwargs)
        return clock.now

    def test_normal_resume_is_bounded(self):
        target = FakeProcess()
        elapsed = self.run_pause(target)
        self.assertEqual(target.signals, [signal.SIGSTOP, signal.SIGCONT])
        self.assertAlmostEqual(elapsed, 2.0)
        self.assertEqual(target.state(), "S")

    def test_already_stopped_target_is_not_touched(self):
        target = FakeProcess("T")
        with self.assertRaises(RuntimeError):
            self.run_pause(target)
        self.assertEqual(target.signals, [])

    def test_invalid_duration_has_no_signals(self):
        # bound raised 2026-10-21 for the Stage-1 SIGSTOP handoff (auto-CONT
        # in finally remains the safety net); CLI clamps 0.05-120.
        for seconds in (0, -1, 120.01, float("nan"), float("inf")):
            target = FakeProcess()
            with self.assertRaises(ValueError):
                self.run_pause(target, seconds=seconds)
            self.assertEqual(target.signals, [])

    def test_missing_stop_ack_still_sends_continue(self):
        target = FakeProcess(acknowledge=False)
        with self.assertRaisesRegex(RuntimeError, "STOP was not acknowledged"):
            self.run_pause(target)
        self.assertEqual(target.signals, [signal.SIGSTOP, signal.SIGCONT])

    def test_post_stop_logging_failure_still_resumes(self):
        target = FakeProcess()

        def broken_log(event, **kwargs):
            if event == "stop_confirmed":
                raise BrokenPipeError("SSH logging lost")

        with self.assertRaises(BrokenPipeError):
            self.run_pause(target, log=broken_log)
        self.assertEqual(target.signals, [signal.SIGSTOP, signal.SIGCONT])
        self.assertEqual(target.state(), "S")

    def test_initial_logging_failure_never_stops(self):
        target = FakeProcess()

        def broken_log(*args, **kwargs):
            raise OSError("output unavailable")

        with self.assertRaises(OSError):
            self.run_pause(target, log=broken_log)
        self.assertEqual(target.signals, [])

    def test_cancel_before_stop_has_no_signals(self):
        target = FakeProcess()
        with self.assertRaisesRegex(RuntimeError, "cancelled before STOP"):
            self.run_pause(target, cancelled=lambda: True)
        self.assertEqual(target.signals, [])

    def test_cancel_during_pause_resumes_early(self):
        target = FakeProcess()
        elapsed = self.run_pause(target, cancelled=lambda: target.state() == "T")
        self.assertEqual(elapsed, 0.0)
        self.assertEqual(target.signals, [signal.SIGSTOP, signal.SIGCONT])

    def test_resume_failure_is_not_reported_as_success(self):
        target = FakeProcess(resume=False)
        with self.assertRaisesRegex(RuntimeError, "CONT did not restore"):
            self.run_pause(target)
        self.assertEqual(target.signals, [signal.SIGSTOP, signal.SIGCONT])


if __name__ == "__main__":
    unittest.main()
