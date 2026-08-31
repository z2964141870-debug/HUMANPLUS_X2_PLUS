import logging
import socket
import struct
import threading
import time

import numpy as np


REQUEST_HEADER = struct.Struct("<4sQI")
RESPONSE_HEADER = struct.Struct("<4sQIf")
REQUEST_MAGIC = b"LFP2"
RESPONSE_MAGIC = b"LFR2"
STATE_NAMES = ("h_1", "c_1", "h_2", "c_2", "h_3", "c_3")
INPUT_SPECS = (("imu_data", (1, 120)),) + tuple(
    (name, (2, 1, 256)) for name in STATE_NAMES
)
OUTPUT_SHAPES = (
    (1, 24, 3),
    (1, 24, 3),
    (1, 24, 3),
    (2, 1, 256),
    (2, 1, 256),
    (2, 1, 256),
    (2, 1, 256),
    (2, 1, 256),
    (2, 1, 256),
)
WIRE_DTYPE = np.dtype("<f4")
REQUEST_FLOATS = sum(int(np.prod(shape)) for _, shape in INPUT_SPECS)
REQUEST_BYTES = REQUEST_FLOATS * WIRE_DTYPE.itemsize
OUTPUT_FLOATS = sum(int(np.prod(shape)) for shape in OUTPUT_SHAPES)
OUTPUT_BYTES = OUTPUT_FLOATS * WIRE_DTYPE.itemsize


def _recv_exact(sock, size):
    payload = bytearray(size)
    view = memoryview(payload)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:])
        if count == 0:
            raise ConnectionError("SoC2 LFP connection closed")
        received += count
    return payload


def _encode_inputs(input_feed):
    if input_feed is None:
        raise ValueError("LFP input_feed is required")
    arrays = []
    for name, shape in INPUT_SPECS:
        if name not in input_feed:
            raise ValueError(f"LFP input_feed is missing {name}")
        value = np.asarray(input_feed[name], dtype=np.float32)
        if value.size != int(np.prod(shape)):
            raise ValueError(
                f"invalid LFP input {name} shape={value.shape}; expected {shape}"
            )
        arrays.append(np.ascontiguousarray(value.reshape(shape), dtype=WIRE_DTYPE))
    return b"".join(value.tobytes() for value in arrays)


def _decode_outputs(payload):
    if len(payload) != OUTPUT_BYTES:
        raise RuntimeError(
            f"invalid SoC2 LFP output size={len(payload)} expected={OUTPUT_BYTES}"
        )
    flat = np.frombuffer(payload, dtype=WIRE_DTYPE)
    outputs = []
    offset = 0
    for shape in OUTPUT_SHAPES:
        size = int(np.prod(shape))
        outputs.append(
            np.asarray(flat[offset : offset + size], dtype=np.float32)
            .reshape(shape)
            .copy()
        )
        offset += size
    return outputs


class RemoteLFPSession:
    """ONNX Session-compatible client with local fallback and remote recovery."""

    def __init__(
        self,
        host,
        port,
        timeout_s,
        local_model,
        local_providers,
        reconnect_interval_s=2.0,
        local_session_factory=None,
    ):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.local_model = local_model
        self.local_providers = local_providers
        self.reconnect_interval_s = max(0.0, reconnect_interval_s)
        self.local_session_factory = local_session_factory
        self.sock = None
        self.sequence = 0
        self.local_session = None
        self.next_remote_attempt = 0.0
        self._remote_active = False
        self._lock = threading.Lock()

    def _connect(self):
        sock = socket.create_connection(
            (self.host, self.port), timeout=self.timeout_s
        )
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self.timeout_s)
        except Exception:
            sock.close()
            raise
        self.sock = sock
        logging.info(
            "LFP backend=SoC2 TCP %s:%d timeout=%.0fms",
            self.host,
            self.port,
            self.timeout_s * 1000.0,
        )

    def _close_remote(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self._remote_active = False

    def close(self):
        with self._lock:
            self._close_remote()

    def _ensure_local_session(self):
        if self.local_session is not None:
            return self.local_session
        if self.local_session_factory is not None:
            self.local_session = self.local_session_factory()
        else:
            import onnxruntime as rt

            self.local_session = rt.InferenceSession(
                self.local_model, providers=self.local_providers
            )
        return self.local_session

    def _remote_run(self, payload):
        if self.sock is None:
            self._connect()
        self.sequence += 1
        sequence = self.sequence
        self.sock.sendall(
            REQUEST_HEADER.pack(REQUEST_MAGIC, sequence, len(payload)) + payload
        )

        header = _recv_exact(self.sock, RESPONSE_HEADER.size)
        magic, response_sequence, payload_size, _remote_ms = RESPONSE_HEADER.unpack(
            header
        )
        if magic != RESPONSE_MAGIC:
            raise RuntimeError(f"invalid SoC2 LFP response magic={magic!r}")
        if response_sequence != sequence:
            raise RuntimeError(
                f"invalid SoC2 LFP response seq={response_sequence} expected={sequence}"
            )
        if payload_size != OUTPUT_BYTES:
            raise RuntimeError(
                f"invalid SoC2 LFP response size={payload_size} expected={OUTPUT_BYTES}"
            )
        outputs = _decode_outputs(_recv_exact(self.sock, payload_size))
        if not self._remote_active:
            logging.info("LFP backend recovered: SoC2 TCP")
        self._remote_active = True
        return outputs

    def run(self, output_names=None, input_feed=None):
        if output_names is not None:
            raise ValueError("RemoteLFPSession currently requires output_names=None")
        payload = _encode_inputs(input_feed)

        # Prevent request/response frames from interleaving if callers overlap.
        with self._lock:
            now = time.monotonic()
            if self.sock is not None or now >= self.next_remote_attempt:
                try:
                    return self._remote_run(payload)
                except (OSError, ConnectionError, RuntimeError) as exc:
                    self._close_remote()
                    self.next_remote_attempt = (
                        time.monotonic() + self.reconnect_interval_s
                    )
                    logging.warning(
                        "SoC2 LFP unavailable; using local ONNX for this frame: %s",
                        exc,
                    )

            # Each request contains the caller's current hidden state, so local
            # fallback and later remote recovery continue the same LSTM stream.
            return self._ensure_local_session().run(output_names, input_feed)
