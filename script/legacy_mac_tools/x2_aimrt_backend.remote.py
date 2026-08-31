#!/usr/bin/env python3
"""X2 AIMRT 真机遥操后端 (新增独立文件, 协议参考板载 x2_mocap_gmr.py 只读得来)

发送: /aima/mc/joint/retargeting (ros2:std_msgs/msg/Float64MultiArray, json, UDP)
62 维布局: [0:7]=root(wxyz后) [7:38]=31身体关节 [38:58]=默认0 [58:60]=claw [60:62]=actuator_type
第一版安全策略: root 用官方默认 (机器人站原地, 只跟关节)
"""
import json
import socket
import struct
import urllib.parse
from typing import Optional

import numpy as np

AIMRT_DEFAULT_IP = "10.0.1.40"
AIMRT_DEFAULT_PORT = 50040
AIMRT_TOPIC = "/aima/mc/joint/retargeting"
AIMRT_MSG_TYPE = "ros2:std_msgs/msg/Float64MultiArray"

OFFICIAL_X2_DEFAULT_QPOS = np.zeros(62, dtype=np.float64)
OFFICIAL_X2_DEFAULT_QPOS[0:7] = [0.0, 0.0, 0.65, 1.0, 0.0, 0.0, 0.0]
OFFICIAL_X2_DEFAULT_QPOS[7:38] = [
    -0.2480, 0.0, 0.0, 0.5303, -0.2823, 0.0,
    -0.2480, 0.0, 0.0, 0.5303, -0.2823, 0.0,
    0.0, 0.0, 0.0,
    0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0,
    0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0,
    0.0, 0.0,
]
OFFICIAL_X2_DEFAULT_QPOS[60:62] = [1.0, 1.0]


def _url_encode(text: str) -> str:
    return urllib.parse.quote(text, safe="")


def _pack_string_uint8(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("B", len(b)) + b


def _pack_string_uint16(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<H", len(b)) + b


def _float64_multi_array_json(values) -> bytes:
    msg = {"layout": {"dim": [], "data_offset": 0},
           "data": np.asarray(values, dtype=np.float64).reshape(-1).tolist()}
    return json.dumps(msg, separators=(",", ":")).encode("utf-8")


def pack_official_qpos(targets31: np.ndarray, root7: Optional[np.ndarray] = None,
                       clamp_limits=None) -> np.ndarray:
    """31 维关节目标 -> 62 维官方 qpos。root 默认官方站姿 (安全第一版: 机器人不位移)。"""
    out = OFFICIAL_X2_DEFAULT_QPOS.copy()
    if root7 is not None:
        q = np.asarray(root7, dtype=np.float64)
        if q.shape == (7,):
            out[0:7] = q
    t = np.asarray(targets31, dtype=np.float64).reshape(-1)
    if t.shape[0] > 31:
        t = t[:31]
    if clamp_limits is not None:
        lo, hi = clamp_limits
        t = np.clip(t, lo, hi)
    out[7:7 + t.shape[0]] = t
    return out


def publish_aimrt_udp(qpos62: np.ndarray, server_ip: str, server_port: int,
                      dry_run: bool = True, *, allow_network: bool = False) -> bool:
    pattern = f"/channel/{_url_encode(AIMRT_TOPIC)}/{_url_encode(AIMRT_MSG_TYPE)}"
    data = _pack_string_uint8(pattern)
    data += _pack_string_uint8("json")
    data += struct.pack("B", 0)  # context_meta 数量
    data += _float64_multi_array_json(qpos62)
    if dry_run:
        print(f"[aimrt-dry] topic={AIMRT_TOPIC} len={len(data)}B qpos[7:11]={np.round(qpos62[7:11],3)}")
        return True
    if not allow_network:
        raise PermissionError(
            "AIMRT network publishing is locked; explicit real-control authorization is required"
        )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(data, (server_ip, int(server_port)))
        return True
    finally:
        sock.close()


class AimrtTeleopBackend:
    """真机遥操后端: send(targets31) -> AIMRT UDP。"""

    def __init__(self, ip: str = AIMRT_DEFAULT_IP, port: int = AIMRT_DEFAULT_PORT,
                 dry_run: bool = True, clamp_limits=None, *,
                 enable_network: bool = False):
        if not dry_run and not enable_network:
            raise PermissionError(
                "real-control backend requested without enable_network=True"
            )
        self.ip = ip
        self.port = port
        self.dry_run = dry_run
        self.enable_network = enable_network
        self.clamp_limits = clamp_limits
        self.last_qpos62 = OFFICIAL_X2_DEFAULT_QPOS.copy()

    def send(self, targets31: np.ndarray, root7: Optional[np.ndarray] = None) -> bool:
        qpos62 = pack_official_qpos(targets31, root7=root7, clamp_limits=self.clamp_limits)
        self.last_qpos62 = qpos62
        return publish_aimrt_udp(
            qpos62, self.ip, self.port, dry_run=self.dry_run,
            allow_network=self.enable_network,
        )
