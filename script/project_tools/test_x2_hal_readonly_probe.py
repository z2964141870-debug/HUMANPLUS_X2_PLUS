import unittest

from x2_hal_readonly_probe import Stream, received_fresh


class StreamTests(unittest.TestCase):
    def test_window_rate_is_not_cumulative(self):
        stream = Stream()
        for now in (0.1, 0.2, 0.3):
            stream.observe(now)
        first = stream.report(1.0, 1.0)
        stream.observe(1.5)
        second = stream.report(2.0, 1.0)
        self.assertEqual(first["rate_hz"], 3.0)
        self.assertEqual(second["rate_hz"], 1.0)
        self.assertEqual(second["total"], 4)

    def test_actual_elapsed_and_monotonic_age(self):
        stream = Stream()
        stream.observe(10.0, 1, 123)
        stream.observe(10.2, 2, 456, True)
        report = stream.report(10.5, 0.5)
        self.assertEqual(report["rate_hz"], 4.0)
        self.assertAlmostEqual(report["age_s"], 0.3)
        self.assertAlmostEqual(report["max_rx_gap_s"], 0.2)
        self.assertEqual(report["sequence_changes"], 1)
        self.assertEqual(report["last_header_ns"], 456)
        self.assertEqual(report["nonzero_commands"], 1)

    def test_ready_requires_every_stream_receiving_now(self):
        streams = {"leg": Stream(), "imu": Stream()}
        streams["leg"].observe(1.0)
        streams["leg"].observe(1.01)
        self.assertFalse(received_fresh(streams, ("leg", "imu"), 1.02))
        streams["imu"].observe(1.0)
        streams["imu"].observe(1.01)
        self.assertTrue(received_fresh(streams, ("leg", "imu"), 1.02))
        self.assertFalse(received_fresh(streams, ("leg", "imu"), 1.2))


if __name__ == "__main__":
    unittest.main()
