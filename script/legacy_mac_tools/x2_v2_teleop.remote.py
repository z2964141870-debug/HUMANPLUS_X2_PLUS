#!/usr/bin/env python3
"""X2 板载 Sonic -> HAL v2 遥操主程序（独立于原 AIMRT 版本）。

链路: 衣服 BLE(V2) -> 校准 -> LFP/SMPL-X(fast) -> GMR(tgt=agibot_x2) -> X2 31参考
      -> Sonic 策略 50 Hz -> 悬吊锚定/安全门控 -> HAL 插值 500 Hz -> 真机
处理链与安全函数 (normalize_mocap_prediction / knee_bias) 对齐板载 x2_mocap_gmr.py (只读复用)
控制台: [1] HAL停发 [2] 悬吊渐增刚度 [3] Sonic悬吊接管 [e] 急停 [q] 退出

运行 (板载):
  cd ~/mocap_teleop/x2_pc1_ble_teleop && conda activate teleop
  python ~/x2_v2_teleop/x2_v2_teleop.py [--dry-run] [--start-mode rc|suit]
"""
import argparse
import atexit
import ctypes
import os
import select
import sys
import time

# Keep the Jetson CUDA stack isolated from the existing teleop environment.
# The overlay contains Jetson/JetPack-matched NumPy, PyTorch and ONNX Runtime
# wheels; when it is absent the original environment is left untouched.
CUDA_OVERLAY_DIR = os.environ.get(
    "X2_CUDA_OVERLAY",
    os.path.expanduser("~/x2_v2_teleop/cuda126_overlay"),
)
if os.path.isdir(CUDA_OVERLAY_DIR):
    sys.path.insert(0, CUDA_OVERLAY_DIR)
    _cupti_dir = os.path.join(
        sys.prefix,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
        "triton",
        "backends",
        "nvidia",
        "lib",
        "cupti",
    )
    _cupti_file = os.path.join(_cupti_dir, "libcupti.so.12")
    _cuda_lib_dirs = [
        _cupti_dir,
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/aarch64-linux/lib",
        "/usr/lib/aarch64-linux-gnu/nvidia",
        "/usr/lib/aarch64-linux-gnu",
    ]
    _old_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    _ld_library_parts = [
        p for p in _cuda_lib_dirs + _old_ld_library_path.split(os.pathsep)
        if p and os.path.isdir(p)
    ]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        dict.fromkeys(_ld_library_parts)
    )
    # LD_LIBRARY_PATH is read by the dynamic loader at process start.  Torch
    # imports libcupti while importing torch._C, so load the bundled CUPTI
    # library explicitly before importing NumPy/Torch/ONNX Runtime.
    if os.path.isfile(_cupti_file):
        try:
            ctypes.CDLL(_cupti_file, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # Keep the normal CPU fallback path available if this optional
            # profiling library is not loadable on a particular image.
            pass

import numpy as np

SUIT_DIR = os.path.expanduser("~/mocap_teleop/x2_pc1_ble_teleop")
LIB_DIR = os.path.expanduser("~/x2_v2_teleop/lib")
# lib 置最前: 确保 v2 hardware_boot 模块优先于板载 V1 (不覆盖板载, 各自独立)
for p in (LIB_DIR, SUIT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from Socket.UDP import *  # noqa: E402,F403  (先星号导入, 防止 utils.config 污染)
import config  # noqa: E402  (星号导入后重新绑定为项目 config)
from general_motion_retargeting.SmpleXConverter import *  # noqa: E402,F403
from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from hardware_boot_v2.receive_bleak_085_v2 import (  # noqa: E402
    BleV2Manager, data_up_buffer, data_down_buffer)

from eval_official_sonic_x2 import (  # noqa: E402
    SonicPolicy, DEFAULT_ANGLES_MJ, JOINT_NAMES, MJ_TO_IL,
)
from x2_aimrt_backend import OFFICIAL_X2_DEFAULT_QPOS  # noqa: E402
from x2_hal_backend import (  # noqa: E402
    HalTeleopBackend, HOLD_LOWER_LIMITS, HOLD_UPPER_LIMITS,
)
from x2_state_feedback import FeedbackReceiver  # noqa: E402
from x2_control_safety import (  # noqa: E402
    ClosedLoopSafetyFilter, SafetyConfig,
)

import onnxruntime as rt  # noqa: E402
import torch  # noqa: E402


def ort_gpu_first_providers():
    """Request CUDA first while retaining a safe CPU fallback.

    A Jetson image can expose the CUDA provider even while its GPU driver is
    temporarily unavailable.  ONNX Runtime then falls back to CPU; printing
    the session providers makes that state visible instead of implying that
    a CUDA provider was actually used.
    """
    available = rt.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return [
            ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": "0"}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]

MODE_RC, MODE_ARMED, MODE_SUIT = 1, 2, 3
MODE_NAME = {
    MODE_RC: "HAL 停发",
    MODE_ARMED: "悬吊当前姿态渐增刚度",
    MODE_SUIT: "衣服v2遥操（悬吊锚定）",
}
LOCAL_X2_LEFT_KNEE_QPOS_IDX, LOCAL_X2_RIGHT_KNEE_QPOS_IDX = 10, 16
LOCAL_X2_KNEE_RANGE = (0.0, 2.4073)


def rebase_suspended_policy_target(policy_targets, anchor, *, scale,
                                    control="arms"):
    """Keep Sonic residuals around the measured load-bearing handoff pose.

    Sonic emits absolute targets around its trained Stand default. Applying
    those absolute targets to a robot carried by a hoist creates an immediate
    tug toward the unsupported training posture. In suspended mode only the
    selected body group receives the bounded residual; all other joints remain
    at the handoff anchor. Head joints are always left at the anchor because
    HAL head command is unavailable on this robot.
    """
    targets = np.asarray(policy_targets, dtype=np.float64).reshape(-1)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if targets.shape != (31,) or anchor.shape != (31,):
        raise ValueError("policy_targets and anchor must have shape (31,)")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(anchor)):
        raise ValueError("policy_targets and anchor must be finite")
    if control not in ("arms", "body"):
        raise ValueError(f"unsupported suspended control group: {control}")
    scale = float(scale)
    if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("scale must be finite and in [0, 1]")

    out = anchor.copy()
    residual = scale * (targets - DEFAULT_ANGLES_MJ)
    if control == "body":
        out[:29] = anchor[:29] + residual[:29]
    else:
        # First load-bearing suspended trial: keep legs and waist fixed and
        # let Sonic exercise only the arms. Full-body release is opt-in.
        out[15:29] = anchor[15:29] + residual[15:29]
    return out


def normalize_mocap_prediction(prediction, server):
    """对齐板载: LFP 输出 -> 24关节 axis_angles + root_translation"""
    if isinstance(prediction, dict):
        axis_angles = prediction.get("axis_angles")
        root_translation = prediction.get("root_translation",
                                          getattr(server, "trans", [0.0, 0.0, 0.0]))
    else:
        axis_angles, root_translation = prediction, getattr(server, "trans", [0.0, 0.0, 0.0])
    for n in ("axis_angles", "root_translation"):
        pass
    if hasattr(axis_angles, "detach"):
        axis_angles = axis_angles.detach().cpu().numpy()
    if hasattr(root_translation, "detach"):
        root_translation = root_translation.detach().cpu().numpy()
    axis_angles = np.asarray(axis_angles, dtype=np.float32).reshape(-1, 3)
    if axis_angles.shape[0] < 24:
        raise ValueError(f"axis_angles too short: {axis_angles.shape}")
    root_translation = np.asarray(root_translation, dtype=np.float32).reshape(-1)
    if root_translation.shape[0] < 3:
        root_translation = np.zeros(3, dtype=np.float32)
    return {"axis_angles": axis_angles[:24], "root_translation": root_translation[:3]}


def apply_lower_body_stability_bias(qpos, knee_bias=0.0):
    """对齐板载: 膝关节稳定偏置 (clip 到 X2 限位)"""
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    if abs(float(knee_bias)) < 1e-9:
        return qpos
    for idx in (LOCAL_X2_LEFT_KNEE_QPOS_IDX, LOCAL_X2_RIGHT_KNEE_QPOS_IDX):
        qpos[idx] = np.clip(qpos[idx] + float(knee_bias), *LOCAL_X2_KNEE_RANGE)
    return qpos


class LiveRef:
    """X2 31 维参考滚动缓冲 (v2 tokenizer 接口)"""
    def __init__(self, max_frames=600):
        self.fps = 50.0
        self.max_frames = max_frames
        self.joint_pos = np.zeros((0, 31), np.float64)
        self.root_quat = np.zeros((0, 4), np.float64)
        self.root_pos = np.zeros((0, 3), np.float64)
        self.last_update = 0.0
        self.name = "suit_live"

    def append(self, jp, rq, rp):
        n = np.linalg.norm(rq)
        rq = rq / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])
        self.joint_pos = np.vstack([self.joint_pos, jp[None, :]])
        self.root_quat = np.vstack([self.root_quat, rq[None, :]])
        self.root_pos = np.vstack([self.root_pos, rp[None, :]])
        if len(self.joint_pos) > self.max_frames:
            self.joint_pos = self.joint_pos[-self.max_frames:]
            self.root_quat = self.root_quat[-self.max_frames:]
            self.root_pos = self.root_pos[-self.max_frames:]
        self.last_update = time.time()

    @property
    def frames(self):
        return len(self.joint_pos)

    def fresh(self, timeout=0.6):
        return (time.time() - self.last_update) < timeout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ble-up", default="D5:F4:A2:41:93:4B")
    ap.add_argument("--ble-down", default="F3:FB:AD:FC:7D:82")
    ap.add_argument("--pose-scale", type=float, default=0.4)
    ap.add_argument("--knee-bias", type=float, default=0.08)
    ap.add_argument("--dry-run", action="store_true",
                    help="兼容参数；程序默认就是演练模式，不发送网络控制包")
    ap.add_argument("--enable-real-control", action="store_true",
                    help="显式解锁 HAL 真机发送（必须同时提供确认短语）")
    ap.add_argument("--real-control-confirm", default="",
                    help=argparse.SUPPRESS)
    ap.add_argument("--allow-open-loop", action="store_true",
                    help="仅允许在 --dry-run 下使用旧的合成 proprioception 路径")
    ap.add_argument("--feedback-bind", default="127.0.0.1",
                    help="真实状态 UDP 监听地址")
    ap.add_argument("--feedback-port", type=int, default=50041,
                    help="真实状态 UDP 监听端口")
    ap.add_argument("--feedback-timeout", type=float, default=0.15,
                    help="真实状态最大允许延迟（秒）")
    ap.add_argument("--max-target-rate", type=float, default=0.20,
                    help="Sonic目标最大变化速度(rad/s)，悬吊首测默认0.20")
    ap.add_argument("--max-policy-offset", type=float, default=0.20,
                    help="相对悬吊锚定姿态的最大关节偏移(rad)")
    ap.add_argument("--max-tracking-error", type=float, default=0.08,
                    help="进入悬吊Sonic前的保持误差门限(rad)")
    ap.add_argument("--suspended-max-tracking-error", type=float, default=0.10,
                    help="悬吊Sonic运行时的最大关节跟踪误差(rad)")
    ap.add_argument("--hal-watchdog", type=float, default=0.50,
                    help="主循环失联后HAL停发延迟(秒)，默认0.50")
    ap.add_argument("--stiffen-seconds", type=float, default=2.0,
                    help="悬吊当前姿态增益从35%%渐入到目标增益的时间")
    ap.add_argument("--suspended-hold-gain", type=float, default=0.65,
                    help="悬吊待命最终增益；承重首测不使用官方Stand增益")
    ap.add_argument("--suspended-sonic-gain", type=float, default=0.55,
                    help="Sonic悬吊接管增益")
    ap.add_argument("--suspended-policy-scale", type=float, default=0.35,
                    help="Sonic残差相对悬吊锚点的缩放")
    ap.add_argument("--suspended-control", choices=["arms", "body"], default="arms",
                    help="悬吊首测控制范围；默认仅双臂，body为显式全身放开")
    ap.add_argument("--suspended-stable-seconds", type=float, default=1.0,
                    help="进入Sonic前要求当前悬吊姿态稳定的时间")
    ap.add_argument("--suspended-max-velocity", type=float, default=0.20,
                    help="进入Sonic前允许的最大关节速度(rad/s)")
    # Kept for command-line compatibility with the previous Stand prototype;
    # the suspended path deliberately does not use either value.
    ap.add_argument("--stand-target-rate", type=float, default=0.15,
                    help=argparse.SUPPRESS)
    ap.add_argument("--stand-tolerance", type=float, default=0.12,
                    help=argparse.SUPPRESS)
    ap.add_argument("--start-mode", choices=["rc", "suit"], default="rc")
    ap.add_argument("--stand-seconds", type=float, default=0.0,
                    help="仅演练模式使用的启动目标计算时长；真机模式禁止启动即发送")
    ap.add_argument("--require-cuda", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="要求 Sonic ONNX session 必须使用 CUDA；默认允许驱动异常时回退 CPU")
    ap.add_argument("--model", default=os.path.expanduser(
        "~/x2_v2_teleop/models/x2_sonic_frozen_g1core_lora_v2.onnx"))
    args = ap.parse_args()

    if args.enable_real_control and args.dry_run:
        ap.error("--dry-run 与 --enable-real-control 不能同时使用")
    if args.enable_real_control and args.real_control_confirm != "X2_SUSPENDED_SONIC_HAL":
        ap.error("真机控制仍锁定：缺少正确的 --real-control-confirm")
    if args.enable_real_control and args.stand_seconds > 0:
        ap.error("真机模式禁止启动即发送站立目标；请保持 --stand-seconds 0")
    if args.enable_real_control and args.start_mode == "suit":
        ap.error("真机模式禁止从 suit 启动；必须从 rc 经过 [2] 待命和 [3] 接管")
    dry_run = not args.enable_real_control
    if args.allow_open_loop and not dry_run:
        ap.error("--allow-open-loop 只允许和 --dry-run 一起使用")
    if not 0.05 <= args.max_target_rate <= 0.50:
        ap.error("--max-target-rate 必须在 0.05 到 0.50 rad/s")
    if not 0.05 <= args.max_policy_offset <= 0.30:
        ap.error("--max-policy-offset 必须在 0.05 到 0.30 rad")
    if not 0.03 <= args.max_tracking_error <= 0.12:
        ap.error("--max-tracking-error 必须在 0.03 到 0.12 rad")
    if not 0.05 <= args.suspended_max_tracking_error <= 0.15:
        ap.error("--suspended-max-tracking-error 必须在 0.05 到 0.15 rad")
    if not 0.25 <= args.hal_watchdog <= 0.75:
        ap.error("--hal-watchdog 必须在 0.25 到 0.75 秒")
    if not 1.0 <= args.stiffen_seconds <= 5.0:
        ap.error("--stiffen-seconds 必须在 1.0 到 5.0 秒")
    if not 0.35 <= args.suspended_hold_gain <= 1.0:
        ap.error("--suspended-hold-gain 必须在 0.35 到 1.0")
    if not 0.20 <= args.suspended_sonic_gain <= 1.0:
        ap.error("--suspended-sonic-gain 必须在 0.20 到 1.0")
    if not 0.05 <= args.suspended_policy_scale <= 1.0:
        ap.error("--suspended-policy-scale 必须在 0.05 到 1.0")
    if not 0.20 <= args.suspended_stable_seconds <= 5.0:
        ap.error("--suspended-stable-seconds 必须在 0.20 到 5.0 秒")
    if not 0.05 <= args.suspended_max_velocity <= 0.50:
        ap.error("--suspended-max-velocity 必须在 0.05 到 0.50 rad/s")

    policy = SonicPolicy(args.model, require_cuda=args.require_cuda)
    print(f"[v2] Sonic ONNX providers: {policy.session.get_providers()}")
    teleop = HalTeleopBackend(
        dry_run=dry_run, enable_network=args.enable_real_control,
        publish_hz=500.0, policy_hz=50.0, watchdog_s=args.hal_watchdog,
    )
    atexit.register(teleop.close)
    feedback = None if args.allow_open_loop else FeedbackReceiver(
        bind_host=args.feedback_bind, port=args.feedback_port)
    safety = ClosedLoopSafetyFilter(SafetyConfig(
        feedback_timeout_s=args.feedback_timeout,
        max_target_rate_rad_s=args.max_target_rate,
        max_policy_offset_rad=args.max_policy_offset,
        # The hold gate remains at --max-tracking-error.  Once Sonic is
        # enabled, a hoist-supported robot can have a small bounded static
        # tracking offset; keep a separate hard ceiling for that mode.
        max_tracking_error_rad=args.suspended_max_tracking_error,
    ))
    print("[v2-hal] Sonic 50Hz -> HAL 500Hz"
          + (" (DRY-RUN，HAL 发布已锁定)" if dry_run else " (悬吊真机发送已显式解锁)"))
    if feedback is None:
        print("[v2] WARNING: dry-run open-loop proprioception enabled")
    else:
        print(f"[v2] 等待真实状态: udp://{args.feedback_bind}:{args.feedback_port} "
              f"(timeout={args.feedback_timeout:.3f}s)")

    if args.stand_seconds > 0:
        print(f"[v2] 安全站立 {args.stand_seconds}s (扶机)...")
        for i in range(int(args.stand_seconds * 50)):
            ramp = min(1.0, i / 30.0)
            q = OFFICIAL_X2_DEFAULT_QPOS.copy()
            q[7:38] = ramp * OFFICIAL_X2_DEFAULT_QPOS[7:38]
            teleop.send(q[7:38], root7=q[0:7])
            time.sleep(1 / 50.0)

    print(f"[v2] 连接衣服 BLE: 上 {args.ble_up} / 下 {args.ble_down}")
    manager = BleV2Manager(addr_up=args.ble_up, addr_down=args.ble_down)
    manager.connect_devices(max_devices=2)

    my_server = DataProcessServer_FullBody(
        rotation_type="AXIS_ANGLE", part="body",
        config=[config.device_config.jacket_6IMU,
                config.device_config.pants_5IMU],
        mode=demo_mode.FULL, track_trans=True,
        calibration_session=None, run_unity_package=False,
        physics_optim=False, cali_pose="T", beta=None)
    retarget = GMR(src_human="smplx", tgt_robot="agibot_x2",
                   actual_human_height=1.8, damping=1.0, verbose=False)
    retarget.max_iter = 1
    smplex = SMPLXConverter()
    lfp = rt.InferenceSession(
        os.path.join(SUIT_DIR, "onnx_models/LFP_dense_taichi_ft.onnx"),
        providers=ort_gpu_first_providers(),
    )
    print(f"[v2] LFP ONNX providers: {lfp.get_providers()}")

    print("[v2] 等待衣服数据并校准... 按 Enter 后保持 T-Pose")
    input()
    while len(data_up_buffer) < 30 or len(data_down_buffer) < 30:
        time.sleep(0.1)
    imu_order_clothes = config.device_config.jacket_6IMU["imu_order"]
    imu_order_pant = config.device_config.pants_5IMU["imu_order"]
    PANT_SLOTS_V2 = [0, 1, 2, 4, 5]
    up = np.array(list(data_up_buffer)[-60:]).reshape(-1, 6, 7)[:, imu_order_clothes, :].mean(axis=0)
    down = np.array(list(data_down_buffer)[-60:]).reshape(-1, 6, 7)[:, PANT_SLOTS_V2, :][:, imu_order_pant, :].mean(axis=0)
    tpose = np.concatenate([up, down], axis=0)  # (10,7)
    tpose_acc = torch.FloatTensor(tpose[:, 0:3])
    tpose_oris = quaternion_to_rotation_matrix(torch.FloatTensor(tpose[:, 3:7]))
    payload = (torch.cat([tpose_acc.reshape(-1), tpose_oris.reshape(-1)], dim=0)
               .detach().cpu().numpy().tolist())
    my_server.set_calibrate_data(payload)
    print("[v2] 校准完成, T-Pose 保持 3 秒...")
    time.sleep(3.0)

    ref = LiveRef()
    lfp_state = {}
    mode = MODE_RC if args.start_mode == "rc" else MODE_SUIT
    armed_target = None
    suspended_anchor = None
    stiffen_started = None
    stiffen_ready = False
    hold_stable_since = None
    hold_gain_complete_reported = False
    last_hold_report = 0.0
    last_feedback_report = 0.0
    print(f"[v2] 当前模式: {MODE_NAME[mode]}")
    if dry_run:
        print("  [1] HAL停发 | [2] 悬吊姿态待命 | [3] Sonic悬吊接管 | [e] 急停 | [q] 退出")
    else:
        print("  [1] HAL停发 | [2] 悬吊渐增刚度 | [3] Sonic悬吊接管 | [e] 急停 | [q] 退出")
    t_last, running = 0.0, True
    while running:
        live_feedback = None
        if feedback is not None:
            feedback.poll()
            live_feedback = feedback.snapshot(args.feedback_timeout)

        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if r:
            ch = sys.stdin.read(1)
            if ch == "1" and mode != MODE_RC:
                teleop.stop()
                mode = MODE_RC
                armed_target = None
                suspended_anchor = None
                stiffen_started = None
                stiffen_ready = False
                hold_stable_since = None
                hold_gain_complete_reported = False
                policy.reset()
                safety.reset()
                print(f"[v2] -> {MODE_NAME[mode]}（HAL停发；恢复官方手柄需另行启动MC）")
            elif ch == "2" and mode == MODE_RC:
                if feedback is not None and live_feedback is None:
                    print("[v2] 拒绝进入衣服模式: 尚无新鲜的真实关节状态，保持停发")
                    continue
                measured = (
                    live_feedback.qpos[7:38]
                    if live_feedback is not None
                    else teleop.last_qpos62[7:38]
                )
                below = measured[:29] < HOLD_LOWER_LIMITS[:29]
                above = measured[:29] > HOLD_UPPER_LIMITS[:29]
                if np.any(below | above):
                    bad = np.flatnonzero(below | above).tolist()
                    print(f"[v2] 拒绝HAL待命: 当前关节超硬限位，indices={bad}")
                    continue
                policy.reset()
                teleop.clear_fault()
                suspended_anchor = live_feedback.qpos[7:38].copy() \
                    if live_feedback is not None else teleop.last_qpos62[7:38].copy()
                safety.reset(
                    suspended_anchor,
                    nominal_joints=suspended_anchor,
                )
                if dry_run:
                    mode = MODE_SUIT
                    print(f"[v2] -> {MODE_NAME[mode]}（演练，腿/腰锚定）")
                else:
                    mode = MODE_ARMED
                    armed_target = suspended_anchor.copy()
                    stiffen_started = time.monotonic()
                    stiffen_ready = False
                    hold_stable_since = None
                    hold_gain_complete_reported = False
                    last_hold_report = 0.0
                    # Start the hold immediately.  The following garment/GMR
                    # update can exceed 100 ms on CPU and must not delay the
                    # first HAL heartbeat.
                    try:
                        teleop.send(armed_target, root7=None, gain_scale=0.35)
                    except Exception as exc:
                        teleop.stop()
                        mode = MODE_RC
                        armed_target = None
                        safety.reset()
                        print(f"[v2] HAL待命启动失败: {exc}，保持停发")
                        continue
                    print(
                        f"[v2] -> 悬吊当前姿态渐增刚度（{args.stiffen_seconds:.1f}s，"
                        f"35% -> {args.suspended_hold_gain:.0%}）"
                    )
                    print("[v2] 刚度完成且姿态稳定后，输入 [3] 启动悬吊 Sonic")
            elif ch == "3" and mode == MODE_ARMED:
                if not stiffen_ready:
                    print("[v2] 悬吊姿态尚未稳定，暂不允许启动 Sonic")
                    continue
                if live_feedback is None:
                    teleop.stop()
                    mode = MODE_RC
                    armed_target = None
                    suspended_anchor = None
                    print("[v2] 拒绝启动Sonic: 真实状态已过期，切回停发")
                    continue
                suspended_anchor = live_feedback.qpos[7:38].astype(np.float64).copy()
                mode = MODE_SUIT
                policy.reset()
                safety.reset(suspended_anchor, nominal_joints=suspended_anchor)
                ref = LiveRef()
                print(
                    f"[v2] -> {MODE_NAME[mode]}（残差缩放 {args.suspended_policy_scale:.2f}，"
                    f"增益 {args.suspended_sonic_gain:.0%}，控制={args.suspended_control}）"
                )
            elif ch == "4" and mode == MODE_ARMED:
                print("[v2] 承重悬吊模式不进入官方 Stand；姿态稳定后请按 [3] 启动 Sonic")
            elif ch in ("e", "E"):
                teleop.stop()
                print("[v2] *** 急停 (停发) ***")
                running = False
                break
            elif ch == "q":
                teleop.stop()
                print("[v2] 退出 (停发)")
                running = False
                break

        if mode == MODE_SUIT and len(data_up_buffer) and len(data_down_buffer):
            data_clothes = np.array(data_up_buffer[-1]).reshape(6, 7)[imu_order_clothes]
            data_pant = np.array(data_down_buffer[-1]).reshape(6, 7)[PANT_SLOTS_V2][imu_order_pant]
            data = np.concatenate([data_clothes, data_pant], axis=0)  # (10,7)
            accs = torch.FloatTensor(data[:, 0:3])
            oris = quaternion_to_rotation_matrix(torch.FloatTensor(data[:, 3:7]))
            payload = (torch.cat([accs.reshape(-1), oris.reshape(-1)], dim=0)
                       .detach().cpu().numpy().tolist())
            payload = my_server.calibrate(payload)
            my_server.operator(payload)
            result = lfp.run(None, my_server.to_predict_data())
            result = my_server.predict_result(result)
            result = normalize_mocap_prediction(result, my_server)
            smpl_bundle = smplex.convert_axis_angle_to_smpl_bundle_fast(
                result["axis_angles"], result["root_translation"])
            qpos = retarget.retarget(smpl_bundle["human_data"])
            qpos = apply_lower_body_stability_bias(qpos, knee_bias=args.knee_bias)
            x2dof = np.asarray(qpos[7:38] if len(qpos) > 31 else qpos, dtype=np.float64)
            if args.pose_scale != 1.0:
                x2dof = DEFAULT_ANGLES_MJ + args.pose_scale * (x2dof - DEFAULT_ANGLES_MJ)
            rq = np.asarray(qpos[3:7] if len(qpos) > 31 else [1, 0, 0, 0], dtype=np.float64)
            ref.append(x2dof, rq,
                       np.asarray(qpos[0:3] if len(qpos) > 31 else [0, 0, 0.65]))

        if mode == MODE_ARMED:
            if live_feedback is None:
                teleop.stop()
                mode = MODE_RC
                armed_target = None
                suspended_anchor = None
                safety.reset()
                print("[v2] 悬吊待命阶段真实状态超时，切回HAL停发")
                continue
            reason = safety.validate_feedback(live_feedback)
            if reason is not None:
                teleop.stop()
                mode = MODE_RC
                armed_target = None
                suspended_anchor = None
                safety.reset()
                print(f"[v2] 悬吊待命阶段状态异常: {reason}，切回HAL停发")
                continue
            armed_errors = np.abs(live_feedback.qpos[7:36] - armed_target[:29])
            worst_armed = int(np.argmax(armed_errors))
            if float(armed_errors[worst_armed]) > args.max_tracking_error:
                teleop.stop()
                mode = MODE_RC
                armed_target = None
                suspended_anchor = None
                safety.reset()
                print(
                    f"[v2] 悬吊待命姿态偏差超限: {JOINT_NAMES[worst_armed]} "
                    f"error={armed_errors[worst_armed]:.5f}rad，切回HAL停发"
                )
                continue
            feedback.poll()
            live_feedback = feedback.snapshot(args.feedback_timeout)
            if live_feedback is None:
                teleop.stop()
                mode = MODE_RC
                armed_target = None
                suspended_anchor = None
                safety.reset()
                print("[v2] 悬吊待命目标发送前状态已过期，切回HAL停发")
                continue
            stiffen_elapsed = max(0.0, time.monotonic() - stiffen_started)
            ramp = min(1.0, stiffen_elapsed / args.stiffen_seconds)
            gain_scale = 0.35 + (args.suspended_hold_gain - 0.35) * ramp
            try:
                teleop.send(armed_target, root7=None, gain_scale=gain_scale)
            except Exception as exc:
                teleop.stop()
                mode = MODE_RC
                armed_target = None
                suspended_anchor = None
                safety.reset()
                print(f"[v2] 悬吊待命异常: {exc}，已停发；请重新输入[2]待命")
                continue
            now_mono = time.monotonic()
            velocity_max = float(np.max(np.abs(live_feedback.qvel[6:35])))
            if ramp >= 1.0 and not hold_gain_complete_reported:
                hold_gain_complete_reported = True
                print(
                    f"[v2] 悬吊刚度渐入完成（{args.suspended_hold_gain:.0%}）；"
                    "等待姿态和速度稳定"
                )
            stable = (
                ramp >= 1.0
                and float(np.max(armed_errors)) <= args.max_tracking_error
                and velocity_max <= args.suspended_max_velocity
            )
            if stable:
                if hold_stable_since is None:
                    hold_stable_since = now_mono
                elif not stiffen_ready and now_mono - hold_stable_since >= args.suspended_stable_seconds:
                    stiffen_ready = True
                    print("[v2] 悬吊姿态稳定门控通过；输入 [3] 启动 Sonic（腿/腰默认锚定）")
            else:
                hold_stable_since = None
            if not stiffen_ready and now_mono - last_hold_report >= 2.0:
                print(
                    f"[v2] 悬吊稳定等待: worst={JOINT_NAMES[worst_armed]} "
                    f"error={armed_errors[worst_armed]:.3f}rad "
                    f"vel_max={velocity_max:.3f}rad/s"
                )
                last_hold_report = now_mono

        if mode == MODE_SUIT:
            if suspended_anchor is None:
                suspended_anchor = (
                    live_feedback.qpos[7:38].astype(np.float64).copy()
                    if live_feedback is not None
                    else teleop.last_qpos62[7:38].astype(np.float64).copy()
                )
                safety.reset(suspended_anchor, nominal_joints=suspended_anchor)
            if feedback is not None and live_feedback is None:
                teleop.stop()
                mode = MODE_RC
                armed_target = None
                suspended_anchor = None
                policy.reset()
                safety.reset()
                now = time.time()
                if now - last_feedback_report > 1.0:
                    print("[v2] 真实状态超时/未到达，切回HAL停发")
                    last_feedback_report = now
                time.sleep(1 / 50.0)
                continue
            if ref.frames >= 40 and ref.fresh():
                m, f = ref, ref.frames - 1
            else:
                class _S:
                    fps = 50.0; frames = 600; name = "stand"
                    joint_pos = np.tile(DEFAULT_ANGLES_MJ, (600, 1))
                    root_quat = np.tile(np.array([1.0, 0, 0, 0]), (600, 1))
                    root_pos = np.tile(np.array([0, 0, 0.65]), (600, 1))
                m, f = _S(), 0
            if live_feedback is not None:
                qp = live_feedback.qpos
                qv = live_feedback.qvel
            else:
                # This branch is intentionally reachable only with
                # --dry-run --allow-open-loop for offline diagnostics.
                qp = np.zeros(38)
                qp[2] = 0.65
                qp[3:7] = [1, 0, 0, 0]
                qp[7:38] = teleop.last_qpos62[7:38]
                qv = np.zeros(37)
            try:
                _, action = policy.infer(m, f / m.fps, qp, qv)
                # v2 必需: 冻结 6 腕关节
                for nm in ("left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
                           "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"):
                    action[MJ_TO_IL[JOINT_NAMES.index(nm)]] = 0.0
                policy_targets = policy.action_to_targets(action, m, f, wrist_ref=False)
                targets = rebase_suspended_policy_target(
                    policy_targets,
                    suspended_anchor,
                    scale=args.suspended_policy_scale,
                    control=args.suspended_control,
                )
                # LFP/GMR/ONNX can consume more than the feedback timeout on
                # the CPU-only board. Refresh immediately before the safety
                # decision so a valid packet that arrived during inference is
                # used; a genuinely missing or stale bridge still fails closed.
                if feedback is not None:
                    feedback.poll()
                    live_feedback = feedback.snapshot(args.feedback_timeout)
                    if live_feedback is None:
                        teleop.stop()
                        mode = MODE_RC
                        armed_target = None
                        suspended_anchor = None
                        policy.reset()
                        safety.reset()
                        print("[v2] 策略计算后真实状态过期，切回HAL停发")
                        continue
                    qp = live_feedback.qpos
                    qv = live_feedback.qvel
                # Head error 1026: do not ask the safety filter to track an
                # actuator this HAL backend intentionally does not command.
                targets[29:31] = qp[36:38]
                decision = (
                    safety.filter_offline(targets, qp[7:38])
                    if feedback is None
                    else safety.filter(
                        targets, live_feedback,
                        enforce_tracking=not dry_run,
                    )
                )
                if not decision.accepted:
                    teleop.stop()
                    mode = MODE_RC
                    armed_target = None
                    suspended_anchor = None
                    policy.reset()
                    detail = f" ({decision.detail})" if decision.detail else ""
                    print(f"[v2] 安全门控拒绝目标: {decision.reason}{detail}，切回HAL停发")
                    continue
                targets = decision.targets
                # Re-check immediately before sending. A stale state must
                # never produce a real command, even if inference was slow.
                if feedback is not None:
                    feedback.poll()
                    if feedback.snapshot(args.feedback_timeout) is None:
                        teleop.stop()
                        mode = MODE_RC
                        armed_target = None
                        suspended_anchor = None
                        policy.reset()
                        safety.reset()
                        print("[v2] 发送前真实状态已过期，切回HAL停发")
                        continue
                teleop.send(
                    targets, root7=None,
                    gain_scale=args.suspended_sonic_gain,
                )
            except Exception as e:
                teleop.stop()
                mode = MODE_RC
                armed_target = None
                suspended_anchor = None
                policy.reset()
                safety.reset()
                print(f"[v2] 策略异常: {e}；切回HAL停发")

        dt = 1 / 50.0
        wait = t_last + dt - time.time()
        if wait > 0:
            time.sleep(wait)
        t_last = time.time()
    if feedback is not None:
        feedback.close()
    teleop.close()
    print("[v2] 已退出")

if __name__ == "__main__":
    main()
