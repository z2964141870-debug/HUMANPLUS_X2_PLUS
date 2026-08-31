import socket
import threading
import time
import unittest

import numpy as np

import lfp_remote
import lfp_worker


def make_feed(marker=0.0):
    feed = {"imu_data": np.full((1, 120), marker, dtype=np.float32)}
    for index, name in enumerate(lfp_remote.STATE_NAMES, start=1):
        feed[name] = np.full((2, 1, 256), marker + index, dtype=np.float32)
    return feed


def make_outputs(feed):
    marker = float(feed["imu_data"][0, 0])
    outputs = [
        np.full(shape, marker + index / 10.0, dtype=np.float32)
        for index, shape in enumerate(lfp_remote.OUTPUT_SHAPES[:3])
    ]
    outputs.extend(np.asarray(feed[name], dtype=np.float32) + 1.0 for name in lfp_remote.STATE_NAMES)
    return outputs


class FakeSession:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def run(self, output_names, input_feed):
        with self.lock:
            self.calls.append(
                {name: np.asarray(value).copy() for name, value in input_feed.items()}
            )
        return make_outputs(input_feed)


class RunningWorker:
    def __init__(self, session=None, timeout_s=0.2, port=0):
        self.session = session or FakeSession()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", port))
        self.server.listen(4)
        self.port = self.server.getsockname()[1]
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=lfp_worker.serve,
            args=(self.server, self.session, timeout_s, self.stop_event),
            daemon=True,
        )

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.server.close()


class RemoteProtocolTests(unittest.TestCase):
    def make_client(self, port, local_session=None, **kwargs):
        local_session = local_session or FakeSession()
        return lfp_remote.RemoteLFPSession(
            "127.0.0.1",
            port,
            timeout_s=kwargs.pop("timeout_s", 0.2),
            local_model="unused.onnx",
            local_providers=["CPUExecutionProvider"],
            reconnect_interval_s=kwargs.pop("reconnect_interval_s", 0.0),
            local_session_factory=lambda: local_session,
            **kwargs,
        )

    def test_wire_round_trip_and_hidden_state_forwarding(self):
        self.assertEqual(lfp_remote.REQUEST_BYTES, lfp_worker.REQUEST_BYTES)
        self.assertEqual(lfp_remote.OUTPUT_BYTES, lfp_worker.OUTPUT_BYTES)
        worker_session = FakeSession()
        with RunningWorker(worker_session) as worker:
            client = self.make_client(worker.port)
            feed = make_feed(7.0)
            outputs = client.run(None, feed)
            client.close()

        self.assertEqual(len(outputs), 9)
        np.testing.assert_array_equal(outputs[3], feed["h_1"] + 1.0)
        self.assertEqual(len(worker_session.calls), 1)
        for name in ("imu_data",) + lfp_remote.STATE_NAMES:
            np.testing.assert_array_equal(worker_session.calls[0][name], feed[name])

    def test_fragmented_tcp_frames(self):
        left, right = socket.socketpair()
        payload = bytes(range(251)) * 17

        def send_fragments():
            for offset in range(0, len(payload), 3):
                right.sendall(payload[offset : offset + 3])
            right.close()

        sender = threading.Thread(target=send_fragments)
        sender.start()
        received = lfp_remote._recv_exact(left, len(payload))
        sender.join(timeout=1.0)
        left.close()
        self.assertEqual(bytes(received), payload)

    def test_timeout_falls_back_locally(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        accepted = threading.Event()

        def blackhole():
            conn, _ = server.accept()
            accepted.set()
            time.sleep(0.25)
            conn.close()
            server.close()

        thread = threading.Thread(target=blackhole, daemon=True)
        thread.start()
        local = FakeSession()
        client = self.make_client(port, local, timeout_s=0.05)
        outputs = client.run(None, make_feed(3.0))
        client.close()
        thread.join(timeout=1.0)

        self.assertTrue(accepted.is_set())
        self.assertEqual(len(local.calls), 1)
        self.assertAlmostEqual(float(outputs[0][0, 0, 0]), 3.0)

    def test_disconnect_fallback_then_remote_reconnect(self):
        first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        first.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        first.bind(("127.0.0.1", 0))
        first.listen(1)
        port = first.getsockname()[1]

        def disconnect_once():
            conn, _ = first.accept()
            conn.close()
            first.close()

        disconnect_thread = threading.Thread(target=disconnect_once, daemon=True)
        disconnect_thread.start()
        local = FakeSession()
        client = self.make_client(port, local, reconnect_interval_s=0.0)
        local_outputs = client.run(None, make_feed(10.0))
        disconnect_thread.join(timeout=1.0)

        worker_session = FakeSession()
        with RunningWorker(worker_session, port=port):
            remote_outputs = client.run(None, make_feed(20.0))
        client.close()

        self.assertEqual(len(local.calls), 1)
        self.assertAlmostEqual(float(local_outputs[0][0, 0, 0]), 10.0)
        self.assertAlmostEqual(float(remote_outputs[0][0, 0, 0]), 20.0)
        self.assertEqual(len(worker_session.calls), 1)

    def test_concurrent_calls_do_not_interleave_frames(self):
        worker_session = FakeSession()
        with RunningWorker(worker_session) as worker:
            client = self.make_client(worker.port)
            results = {}

            def invoke(marker):
                results[marker] = client.run(None, make_feed(marker))

            threads = [threading.Thread(target=invoke, args=(float(i),)) for i in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2.0)
            client.close()

        self.assertEqual(len(results), 8)
        for marker, outputs in results.items():
            self.assertAlmostEqual(float(outputs[0][0, 0, 0]), marker)
        self.assertEqual(len(worker_session.calls), 8)

    def test_two_clients_can_use_worker(self):
        worker_session = FakeSession()
        with RunningWorker(worker_session) as worker:
            clients = [self.make_client(worker.port) for _ in range(2)]
            results = {}

            def invoke(index):
                results[index] = clients[index].run(None, make_feed(float(index)))

            threads = [threading.Thread(target=invoke, args=(index,)) for index in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2.0)
            for client in clients:
                client.close()

        self.assertEqual(len(results), 2)
        self.assertEqual(len(worker_session.calls), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
