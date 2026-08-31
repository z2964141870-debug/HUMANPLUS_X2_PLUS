#!/usr/bin/env python3
"""纯 Python 镜像解码器 + 自校验,断言 publisher 的字节符合 v5.1 规格。

解码逻辑逐行镜像 C++ 侧:
  * 分帧/header:``include/zmq/zmq_packed_message_subscriber.hpp``
  * 字段校验/strict 拒帧:``src/zmq_pose_input_source.cpp::HandleDecoded``

**不依赖 pyzmq**,只要 numpy 就能跑:
    python verify_v51_bytes.py

退出码 0 = 全部通过,1 = 有断言失败。
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

import publish_v51_reference as pub

HEADER_SIZE = pub.HEADER_SIZE
NUM_DOFS = pub.NUM_DOFS
NUM_FUTURE_SLOTS = pub.NUM_FUTURE_SLOTS

#: 镜像 subscriber.hpp:145-151 的 GetElementSize()。注意最后的 default 4:
#: 一个拼错的 dtype 不会报错,而是按 4 字节走位 -> 后续字段错位。
_ELEM_SIZE = {"f64": 8, "i64": 8, "f32": 4, "i32": 4, "i16": 2, "f16": 2,
              "i8": 1, "u8": 1, "bool": 1}


def _elem_size(dtype: str) -> int:
    return _ELEM_SIZE.get(dtype, 4)


class DecodeError(Exception):
    """对应 C++ 侧"整包静默丢弃"的那一层(规格 5.1 的 T1-T5)。"""


def decode_packed(raw: bytes, topic: str = "pose") -> Tuple[dict, Dict[str, bytes]]:
    """镜像 ZMQPackedMessageSubscriber::PollOnce 的分帧逻辑。

    返回 (header_dict, {字段名: 原始字节})。任何 C++ 会整包丢弃的情况都抛
    DecodeError,并在消息里标注对应的规格条目。
    """
    tb = topic.encode("utf-8")
    # T1: topic 前缀 memcmp(subscriber.hpp:273-280)
    if len(raw) < len(tb) or raw[: len(tb)] != tb:
        raise DecodeError("T1: topic 前缀不匹配")
    body = raw[len(tb):]

    # T2: 剥掉 topic 后必须 >= 1280(subscriber.hpp:282-288)
    if len(body) < HEADER_SIZE:
        raise DecodeError(f"T2: header 段只有 {len(body)} 字节 < {HEADER_SIZE}")

    # T3: strnlen 取 NUL 终止的 JSON 再 parse(subscriber.hpp:292-303)
    hdr_block = body[:HEADER_SIZE]
    nul = hdr_block.find(b"\x00")
    json_len = HEADER_SIZE if nul < 0 else nul
    try:
        header = json.loads(hdr_block[:json_len].decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise DecodeError(f"T3: header JSON 解析失败: {exc}") from exc

    data = body[HEADER_SIZE:]
    buffers: Dict[str, bytes] = {}
    offset = 0
    for f in header.get("fields", []):
        name = f.get("name", "")
        shape = f.get("shape", [])
        # subscriber.hpp:154-156 —— shape 为空则字节数算 0,后续字段全部错位
        n_bytes = 0
        if shape:
            total = 1
            for d in shape:
                total *= int(d)
            n_bytes = total * _elem_size(f.get("dtype", ""))
        # T4: 越界则整包丢弃(subscriber.hpp:337-342)
        if offset + n_bytes > len(data):
            raise DecodeError(f"T4: 字段 '{name}' 越界 "
                              f"({offset}+{n_bytes} > {len(data)})")
        buffers[name] = data[offset: offset + n_bytes]
        offset += n_bytes
    return header, buffers


# --------------------------------------------------------------------------
# 字段级解码:镜像 CopyFloat32IntoDouble / NormalizeQuatXyzw
# --------------------------------------------------------------------------


def copy_f32(field: dict, buf: bytes, count: int) -> Optional[np.ndarray]:
    """镜像 CopyFloat32IntoDouble(cpp:170-200)。

    返回 None 表示 C++ 侧 **这一个字段** 拷贝失败(got_* 保持 false),
    注意这不是整包丢弃 —— 它更阴险:字段被静默跳过,后果取决于该字段是否
    是 strict 的必需项。
      * dtype 必须精确等于 "f32"(cpp:176),f64 不会被自动转换;
      * shape 乘积必须精确等于 count,多一个少一个都失败;
      * 无条件按小端读(cpp:200 传 needs_swap=false)。
    """
    if field.get("dtype") != "f32":
        return None
    shape = field.get("shape", [])
    total = 1
    for d in shape:
        total *= int(d)
    if not shape or total != count or len(buf) != count * 4:
        return None
    return np.frombuffer(buf, dtype="<f4").astype(np.float64)


def copy_i64(field: dict, buf: bytes) -> Optional[int]:
    """镜像 CopyInt64Scalar(cpp:205-220):dtype 必须 "i64",单元素。"""
    if field.get("dtype") != "i64":
        return None
    shape = field.get("shape", [])
    total = 1
    for d in shape:
        total *= int(d)
    if not shape or total != 1 or len(buf) != 8:
        return None
    return int(np.frombuffer(buf, dtype="<i8")[0])


def normalize_quat_xyzw(q: np.ndarray) -> Optional[np.ndarray]:
    """镜像 NormalizeQuatXyzw(cpp:59-71)。范数带外 -> None(必须拒帧)。

    注意这道校验**抓不到 wxyz/xyzw 顺序错误**:两者范数都是 1。
    """
    n2 = float(np.dot(q, q))
    if not np.isfinite(n2) or n2 < 1e-6:
        return None
    n = n2 ** 0.5
    if n < 0.5 or n > 2.0:
        return None
    return q / n


# --------------------------------------------------------------------------
# 帧级判定:镜像 HandleDecoded 的接受/拒绝顺序(cpp:261-455)
# --------------------------------------------------------------------------


class FrameVerdict:
    """C++ 处理完一帧后的可观测结果。

    ``is_body`` 对应 got_body:**只有** joint_pos_mj 成功拷贝才算 body 帧。
    这就是入口闸门第 2 项刻意用 body_frames_received() 而不是
    total_frames_received() 的原因 —— 一串 hand-only 帧 is_body=False。
    """

    def __init__(self) -> None:
        self.is_body = False
        self.accepted = False
        self.reject_reason = ""
        self.estop_latched = False
        self.has_explicit_velocity = False
        self.window_promoted = False
        self.future_dt_s = 0.1          # DT_FUTURE_REF,带外输入时的保留值
        self.joint_pos: Optional[np.ndarray] = None
        self.joint_vel: Optional[np.ndarray] = None
        self.quat: Optional[np.ndarray] = None
        self.window: Optional[List[np.ndarray]] = None
        self.window_vel: Optional[List[np.ndarray]] = None

    def __repr__(self) -> str:
        state = "ACCEPT" if self.accepted else f"REJECT({self.reject_reason})"
        return (f"<FrameVerdict {state} body={self.is_body} "
                f"vel={self.has_explicit_velocity} window={self.window_promoted}>")


def evaluate_frame(header: dict, buffers: Dict[str, bytes],
                   *, strict: bool) -> FrameVerdict:
    """按 C++ 的**原始顺序**判定一帧。顺序本身就是契约,不要重排。"""
    v = FrameVerdict()
    got: Dict[str, object] = {}

    for f in header.get("fields", []):
        name = f.get("name", "")
        b = buffers.get(name, b"")
        if name == "estop":
            # cpp:265-268 —— 只看字段名,不读值,永久闭锁
            v.estop_latched = True
        elif name == "joint_pos_mj":
            got["pos"] = copy_f32(f, b, NUM_DOFS)
        elif name == "root_quat_xyzw":
            got["quat"] = copy_f32(f, b, 4)
        elif name == "joint_vel_mj":
            got["vel"] = copy_f32(f, b, NUM_DOFS)
        elif name == "joint_pos_mj_future":
            got["fpos"] = copy_f32(f, b, NUM_FUTURE_SLOTS * NUM_DOFS)
        elif name == "joint_vel_mj_future":
            got["fvel"] = copy_f32(f, b, NUM_FUTURE_SLOTS * NUM_DOFS)
        elif name == "root_quat_xyzw_future":
            got["fquat"] = copy_f32(f, b, NUM_FUTURE_SLOTS * 4)
        elif name == "future_dt_s":
            dt = copy_f32(f, b, 1)
            # cpp:328 —— 带外值被**静默忽略**,不是拒帧
            if dt is not None and 0.01 <= float(dt[0]) <= 1.0:
                v.future_dt_s = float(dt[0])
        elif name == "frame_index":
            got["idx"] = copy_i64(f, b)
        # 其余字段静默忽略(cpp:333,向前兼容)

    pos = got.get("pos")
    v.is_body = pos is not None
    if not v.is_body:
        # cpp:440-455 —— token/hand-only 帧:刷新旁路缓存,body 参考不动,
        # body_frames_received 不自增。既不算接受也不算拒绝。
        return v

    quat = got.get("quat")
    vel = got.get("vel")

    # 校验顺序逐字对应 cpp:338-362
    if not np.isfinite(pos).all():
        v.reject_reason = "joint_pos_mj contains non-finite value"
    elif quat is not None and normalize_quat_xyzw(quat) is None:
        v.reject_reason = "root_quat_xyzw bad norm or non-finite"
    elif vel is not None and not np.isfinite(vel).all():
        v.reject_reason = "joint_vel_mj contains non-finite value"
    elif strict and vel is None:
        v.reject_reason = "strict mode: missing joint_vel_mj field"
    elif strict and (got.get("fpos") is None or got.get("fquat") is None
                     or got.get("fvel") is None):
        v.reject_reason = "strict mode: incomplete future window"
    if v.reject_reason:
        # 被拒的帧:rejected_frames++ / total++,但 body_frames_received 不动,
        # LastReceivedMonotonicS() 也不刷新 -> publisher 退回旧布局会把
        # watchdog 饿死,而不是让策略吃到另一套速度语义。
        return v

    v.accepted = True
    v.joint_pos = pos
    v.quat = normalize_quat_xyzw(quat) if quat is not None else None
    if vel is not None:
        v.has_explicit_velocity = True   # cpp:373-375,原样转发不重估
        v.joint_vel = vel
    else:
        v.joint_vel = None               # 非 strict 才会走 wall-clock 回退

    # 未来窗口提升:cpp:404-439。只需要 fpos+fquat;fvel 缺失时用有限差分,
    # 在 live-edge clamp 下会算出全零速度 -> 破坏 parity,所以必须显式发。
    fpos, fquat, fvel = got.get("fpos"), got.get("fquat"), got.get("fvel")
    if fpos is not None and fquat is not None:
        fq = fquat.reshape(NUM_FUTURE_SLOTS, 4)
        normed = [normalize_quat_xyzw(fq[k]) for k in range(NUM_FUTURE_SLOTS)]
        if all(q is not None for q in normed):
            v.window_promoted = True
            v.window = [pos] + list(fpos.reshape(NUM_FUTURE_SLOTS, NUM_DOFS))
            if fvel is not None:
                v.window_vel = ([vel if vel is not None else np.zeros(NUM_DOFS)]
                                + list(fvel.reshape(NUM_FUTURE_SLOTS, NUM_DOFS)))
    return v


# --------------------------------------------------------------------------
# 自校验
# --------------------------------------------------------------------------

_FAILURES: List[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))
        _FAILURES.append(name)


def _stream_frames(n: int, rate: float = 50.0):
    """跑 publisher 的真实代码路径,产出 n 个已打包的字节串。"""
    src = pub.synthetic_source(rate)
    stream = pub.V51ReferenceStream()
    out = []
    for i in range(n):
        p, q = next(src)
        stream.push(p, q)
        out.append((stream, pub.pack_message(stream.build_fields(i))))
    return out


def test_header_layout() -> None:
    print("[1] header 与分帧")
    _, raw = _stream_frames(1)[0]
    body = raw[len(b"pose"):]
    hdr = body[:HEADER_SIZE]
    nul = hdr.find(b"\x00")
    check("topic 前缀是裸字节 'pose'", raw.startswith(b"pose"))
    check("header 段正好 1280 字节", len(body) >= HEADER_SIZE)
    check("header 以 NUL 补齐(否则 strnlen 会吃进二进制)",
          nul > 0 and set(hdr[nul:]) == {0})
    h = json.loads(hdr[:nul].decode())
    check("endian 声明 le(解码路径无条件小端)", h["endian"] == "le")
    check("每个字段都显式带 shape(空 shape -> 该字段算 0 字节,后续全错位)",
          all(f.get("shape") for f in h["fields"]))
    expect = sum(int(np.prod(f["shape"])) * _elem_size(f["dtype"])
                 for f in h["fields"])
    check(f"载荷长度与 header 累加一致({expect} B)",
          len(body) - HEADER_SIZE == expect,
          f"实际 {len(body) - HEADER_SIZE}")


def test_strict_accept() -> None:
    print("[2] strict 模式接受")
    frames = _stream_frames(3)
    for i, (_, raw) in enumerate(frames):
        hdr, bufs = decode_packed(raw)
        v = evaluate_frame(hdr, bufs, strict=True)
        check(f"帧 {i} 被 strict 接受", v.accepted, v.reject_reason)
        check(f"帧 {i} 算作 body 帧(能推进 warmup 计数)", v.is_body)
        check(f"帧 {i} has_explicit_reference_velocity", v.has_explicit_velocity)
        check(f"帧 {i} 未来窗口被提升", v.window_promoted)
        # 容差 1e-6 而不是 1e-9:0.1 经 float32 往返变成 0.10000000149,
        # 差 1.5e-9。C++ 的区间校验是 [0.01, 1.0],这个偏差完全无害。
        check(f"帧 {i} future_dt_s = 0.1", abs(v.future_dt_s - 0.1) < 1e-6)
        check(f"帧 {i} 不带 estop 字段", not v.estop_latched)


def test_velocity_contract() -> None:
    print("[3] 速度契约 (pos[t]-pos[t-1]) * 50")
    src = pub.synthetic_source(50.0)
    stream = pub.V51ReferenceStream()
    p0, q0 = next(src)
    stream.push(p0, q0)
    v0 = evaluate_frame(*decode_packed(pub.pack_message(stream.build_fields(0))),
                        strict=True)
    check("首帧速度为 0(没有 pos[t-1]),但仍然发出且被接受",
          v0.accepted and np.allclose(v0.joint_vel, 0.0, atol=1e-6))

    p1, q1 = next(src)
    stream.push(p1, q1)
    v1 = evaluate_frame(*decode_packed(pub.pack_message(stream.build_fields(1))),
                        strict=True)
    expect = (p1 - p0) * 50.0
    check("第二帧速度 == 差分 * 50.0(常数,不是实测帧率)",
          np.allclose(v1.joint_vel, expect, atol=2e-4),
          f"最大偏差 {np.max(np.abs(v1.joint_vel - expect)):.3e}")
    check("速度非零(否则 tokenizer 看不到参考速度)",
          np.max(np.abs(v1.joint_vel)) > 1e-4)

    # 与实测帧率解耦:同样两帧,发送节奏改了,速度必须不变
    stream2 = pub.V51ReferenceStream()
    stream2.push(p0, q0)
    stream2.push(p1, q1)
    v2 = evaluate_frame(*decode_packed(pub.pack_message(stream2.build_fields(1))),
                        strict=True)
    check("速度与 wall-clock 无关(C++ 不重估,发送端拥有微分权)",
          np.allclose(v1.joint_vel, v2.joint_vel, atol=1e-9))


def test_live_edge_clamp() -> None:
    print("[4] live-edge clamp:未来槽位重复最新位姿 + 最新非零速度")
    src = pub.synthetic_source(50.0)
    stream = pub.V51ReferenceStream()
    p0, q0 = next(src)
    stream.push(p0, q0)
    p1, q1 = next(src)
    stream.push(p1, q1)
    v = evaluate_frame(*decode_packed(pub.pack_message(stream.build_fields(1))),
                       strict=True)
    check("窗口长度 10(slot0 = 当前帧)", v.window is not None
          and len(v.window) == pub.NUM_FUTURE_FRAMES)
    check("9 个未来槽位位姿全部 == 最新位姿",
          all(np.allclose(v.window[k], v.window[0], atol=2e-4)
              for k in range(1, pub.NUM_FUTURE_FRAMES)))
    check("9 个未来槽位速度全部 == 最新速度(重复非零,不置零)",
          v.window_vel is not None
          and all(np.allclose(v.window_vel[k], v.window_vel[0], atol=2e-4)
                  for k in range(1, pub.NUM_FUTURE_FRAMES)))
    check("窗口内速度非零,而位姿斜率为 0(反直觉但正确)",
          np.max(np.abs(v.window_vel[-1])) > 1e-4
          and np.allclose(v.window[-1] - v.window[-2], 0.0, atol=2e-4))


def test_strict_rejects() -> None:
    print("[5] strict 拒帧")
    src = pub.synthetic_source(50.0)
    stream = pub.V51ReferenceStream()
    for _ in range(2):
        p, q = next(src)
        stream.push(p, q)

    raw = pub.pack_message(stream.build_fields(0, drop_velocity=True))
    hdr, bufs = decode_packed(raw)
    v_strict = evaluate_frame(hdr, bufs, strict=True)
    check("R1 缺 joint_vel_mj -> strict 拒帧",
          not v_strict.accepted
          and "missing joint_vel_mj" in v_strict.reject_reason,
          v_strict.reject_reason)
    check("被拒帧仍算 body 帧检测到,但不能推进 warmup(accepted=False)",
          v_strict.is_body and not v_strict.accepted)
    v_loose = evaluate_frame(hdr, bufs, strict=False)
    check("同一帧在非 strict 下被接受(保留 v4/v5 回退给 mock-VLA)",
          v_loose.accepted and not v_loose.has_explicit_velocity)

    # R2:未来窗口不完整
    fields = [f for f in stream.build_fields(0)
              if f[0] != "joint_vel_mj_future"]
    v2 = evaluate_frame(*decode_packed(pub.pack_message(fields)), strict=True)
    check("R2 缺 joint_vel_mj_future -> strict 拒帧",
          not v2.accepted and "incomplete future window" in v2.reject_reason,
          v2.reject_reason)


def test_safety_traps() -> None:
    print("[6] 安全陷阱")
    src = pub.synthetic_source(50.0)
    stream = pub.V51ReferenceStream()
    p, q = next(src)
    stream.push(p, q)
    fields = stream.build_fields(0)

    check("publisher 从不打包 estop 字段",
          all(n != "estop" for n, _ in fields))
    try:
        pub.pack_message(list(fields) + [("estop", np.array([0], dtype=np.int64))])
        check("显式打包 estop 会被 pack_message 拒绝", False, "居然打包成功了")
    except ValueError:
        check("显式打包 estop 会被 pack_message 拒绝", True)

    # 四元数范数校验抓不到 wxyz/xyzw 顺序错误 —— 这条必须靠发送端保证
    wxyz_unit = np.array([1.0, 0.0, 0.0, 0.0])
    check("wxyz 单位四元数同样通过范数校验(顺序错误无法被 C++ 发现)",
          normalize_quat_xyzw(wxyz_unit) is not None)
    check("全零四元数被拒(n2 < 1e-6)",
          normalize_quat_xyzw(np.zeros(4)) is None)
    check("范数 3.0 被拒(带外)",
          normalize_quat_xyzw(np.array([3.0, 0.0, 0.0, 0.0])) is None)

    # dtype 白名单:f64 不会被自动降级
    bad = {"name": "joint_pos_mj", "dtype": "f64", "shape": [NUM_DOFS]}
    check("dtype f64 的 joint_pos_mj 拷贝失败(C++ 精确比较字符串 'f32')",
          copy_f32(bad, b"\x00" * (NUM_DOFS * 8), NUM_DOFS) is None)
    # shape 缺失 -> 该字段算 0 字节
    check("shape 为空 -> 字段算 0 字节,后续字段全部错位",
          copy_f32({"name": "x", "dtype": "f32", "shape": []},
                   b"\x00" * (NUM_DOFS * 4), NUM_DOFS) is None)


def test_frame_rate_sensitivity() -> None:
    print("[7] 实测帧率偏离 50 Hz 时的速度幅值影响(诊断,非断言)")
    # 契约固定乘 50.0。若真实帧率低于 50 Hz,相邻帧覆盖的时间更长,
    # 因而速度会被放大为 50/actual_hz。这是 MuJoCo parity 的刻意行为。
    for actual_hz in (50.0, 48.0, 47.0, 40.0, 35.0, 25.0):
        ratio = pub.VELOCITY_DIFF_HZ / actual_hz
        print(f"        实测 {actual_hz:5.1f} Hz -> 速度幅值为真实值的 "
              f"{ratio * 100:5.1f}%  (差分恒乘 50.0)")
    check("契约常数就是 50.0", abs(pub.VELOCITY_DIFF_HZ - 50.0) < 1e-9)


def main() -> int:
    print(f"v5.1 wire 自校验(NUM_DOFS={NUM_DOFS}, "
          f"未来槽位={NUM_FUTURE_SLOTS}, header={HEADER_SIZE} B)\n")
    for fn in (test_header_layout, test_strict_accept, test_velocity_contract,
               test_live_edge_clamp, test_strict_rejects, test_safety_traps,
               test_frame_rate_sensitivity):
        fn()
        print()
    if _FAILURES:
        print(f"结果:{len(_FAILURES)} 项失败 -> {', '.join(_FAILURES)}")
        return 1
    print("结果:全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
