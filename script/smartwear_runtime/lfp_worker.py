import argparse
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


def recv_exact(sock, size):
    payload = bytearray(size)
    view = memoryview(payload)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:])
        if count == 0:
            raise ConnectionError("client disconnected")
        received += count
    return payload


def decode_inputs(payload):
    if len(payload) != REQUEST_BYTES:
        raise ValueError(
            f"invalid request size={len(payload)} expected={REQUEST_BYTES}"
        )
    flat = np.frombuffer(payload, dtype=WIRE_DTYPE)
    feed = {}
    offset = 0
    for name, shape in INPUT_SPECS:
        size = int(np.prod(shape))
        feed[name] = (
            np.asarray(flat[offset : offset + size], dtype=np.float32)
            .reshape(shape)
            .copy()
        )
        offset += size
    return feed


def encode_outputs(outputs):
    if len(outputs) != len(OUTPUT_SHAPES):
        raise ValueError(
            f"model returned {len(outputs)} outputs; expected {len(OUTPUT_SHAPES)}"
        )
    payload = []
    for index, (output, shape) in enumerate(zip(outputs, OUTPUT_SHAPES)):
        value = np.asarray(output, dtype=np.float32)
        if value.size != int(np.prod(shape)):
            raise ValueError(
                f"invalid model output {index} shape={value.shape}; expected {shape}"
            )
        payload.append(
            np.ascontiguousarray(value.reshape(shape), dtype=WIRE_DTYPE).tobytes()
        )
    encoded = b"".join(payload)
    if len(encoded) != OUTPUT_BYTES:
        raise AssertionError("internal LFP output framing error")
    return encoded


def make_session(model, threads):
    import onnxruntime as rt

    options = rt.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return rt.InferenceSession(
        model, sess_options=options, providers=["CPUExecutionProvider"]
    )


def serve_client(conn, session, inference_lock=None, timeout_s=1.0):
    conn.settimeout(timeout_s)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        header = recv_exact(conn, REQUEST_HEADER.size)
        magic, sequence, payload_size = REQUEST_HEADER.unpack(header)
        if magic != REQUEST_MAGIC:
            raise ValueError(f"invalid request magic {magic!r}")
        if payload_size != REQUEST_BYTES:
            raise ValueError(
                f"invalid request payload size={payload_size} expected={REQUEST_BYTES}"
            )
        feed = decode_inputs(recv_exact(conn, payload_size))

        start = time.perf_counter()
        if inference_lock is None:
            outputs = session.run(None, feed)
        else:
            with inference_lock:
                outputs = session.run(None, feed)
        infer_ms = (time.perf_counter() - start) * 1000.0
        payload = encode_outputs(outputs)
        conn.sendall(
            RESPONSE_HEADER.pack(
                RESPONSE_MAGIC, sequence, len(payload), infer_ms
            )
            + payload
        )


def _client_thread(conn, address, session, inference_lock, timeout_s):
    print(f"[soc2-lfp] client={address[0]}:{address[1]}", flush=True)
    with conn:
        try:
            serve_client(conn, session, inference_lock, timeout_s)
        except Exception as exc:
            print(f"[soc2-lfp] client closed: {exc}", flush=True)


def serve(server, session, timeout_s=1.0, stop_event=None):
    inference_lock = threading.Lock()
    client_threads = []
    server.settimeout(0.2)
    while stop_event is None or not stop_event.is_set():
        try:
            conn, address = server.accept()
        except socket.timeout:
            continue
        thread = threading.Thread(
            target=_client_thread,
            args=(conn, address, session, inference_lock, timeout_s),
            daemon=True,
        )
        thread.start()
        client_threads.append(thread)
        client_threads = [item for item in client_threads if item.is_alive()]
    for thread in client_threads:
        thread.join(timeout=timeout_s + 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--bind", default="10.0.1.42")
    parser.add_argument("--port", type=int, default=51236)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--client-timeout", type=float, default=1.0)
    args = parser.parse_args()

    session = make_session(args.model, args.threads)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind, args.port))
        server.listen(4)
        print(
            f"[soc2-lfp] READY protocol=LFP2 tcp://{args.bind}:{args.port} "
            f"threads={args.threads} client_timeout={args.client_timeout:.2f}s",
            flush=True,
        )
        serve(server, session, timeout_s=args.client_timeout)


if __name__ == "__main__":
    main()
