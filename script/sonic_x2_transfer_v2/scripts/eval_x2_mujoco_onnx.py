#!/usr/bin/env python3
"""ONNX-driven MuJoCo evaluation for X2 Ultra.

Point it at the fused SONIC ONNX and a motion-lib PKL and it launches a
MuJoCo viewer with the policy tracking the clip; ``--no-viewer`` runs the
same loop headless, ``--kinematic`` plays the reference without physics,
``--record`` captures either mode to mp4.

The fused ONNX takes a single 1670-D vector::

    actor_obs = [tokenizer_obs(680) | proprioception(990)]

and returns a 31-D action in IsaacLab DOF order. Observation construction
(and all X2 constants) come from :mod:`eval_x2_mujoco`, shared with the
deployment stack. The real robot's deploy tuning preset (PD trim, target
clamps, LPF, action clip) is applied by default — see ``--tuning``.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

# Reuse all constants and helpers from eval_x2_mujoco.py — by importing rather
# than copy-pasting, both scripts stay in lockstep if the X2 constants ever
# change (kp/kd/action_scale/joint maps/default angles all derive from the
# same source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_x2_mujoco import (  # noqa: E402  (sys.path setup must come first)
    ACTION_SCALE,
    CONTROL_DT,
    DECIMATION,
    DEFAULT_DOF,
    IL_TO_MJ_DOF,
    JOINT_TO_ACTUATOR,
    KD,
    KP,
    MJ_TO_IL_DOF,
    MJCF_PATH,
    MUJOCO_JOINT_NAMES,
    NUM_DOFS,
    SIM_DT,
    ProprioceptionBuffer,
    build_tokenizer_obs,
    compute_motion_state,
    get_motion_fps,
    get_total_frames,
    load_deploy_tuning,
    load_motion_data,
    quat_rotate_inverse,
)


# Tokenizer layout constants for the X2 g1 encoder (sourced from training
# config gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_ultra*):
#   command_multi_future_nonflat:    (NUM_FUTURE_FRAMES=10, COMMAND_DIM_PER_FRAME=62)
#   motion_anchor_ori_b_mf_nonflat:  (NUM_FUTURE_FRAMES=10, ORI_DIM_PER_FRAME=6)
# Total tokenizer width: 10*62 + 10*6 = 680.
NUM_FUTURE_FRAMES_TOK = 10
COMMAND_DIM_PER_FRAME = 62
ORI_DIM_PER_FRAME = 6
COMMAND_FLAT_DIM = NUM_FUTURE_FRAMES_TOK * COMMAND_DIM_PER_FRAME  # 620
ORI_FLAT_DIM = NUM_FUTURE_FRAMES_TOK * ORI_DIM_PER_FRAME  # 60
TOK_DIM = COMMAND_FLAT_DIM + ORI_FLAT_DIM  # 680
PROP_DIM = 990
ACTOR_OBS_DIM = TOK_DIM + PROP_DIM  # 1670


class SupportedDeployShaper:
    """Exact fixed-reference supported command shaping used by X2 deploy.

    This is intentionally separate from the historical generic simulator path.
    It mirrors the powered supported probe ordering:

        safety soft-start/clamp -> supported ramp/envelope -> LPF -> slew

    The static anchor is supplied explicitly so the first LPF and slew states
    match the command held immediately before policy entry.
    """

    def __init__(
        self,
        default: np.ndarray,
        safety_max_dev: np.ndarray,
        supported_max_dev: np.ndarray,
        lpf_alpha: np.ndarray,
        safety_ramp_seconds: float,
        supported_ramp_seconds: float,
        slew_rate: float,
        dt: float,
    ) -> None:
        self.default = np.asarray(default, dtype=np.float64).copy()
        self.safety_max_dev = np.asarray(
            safety_max_dev, dtype=np.float64
        ).copy()
        self.supported_max_dev = np.asarray(
            supported_max_dev, dtype=np.float64
        ).copy()
        self.lpf_alpha = np.asarray(lpf_alpha, dtype=np.float64).copy()
        self.safety_ramp_seconds = float(safety_ramp_seconds)
        self.supported_ramp_seconds = float(supported_ramp_seconds)
        self.slew_rate = float(slew_rate)
        self.dt = float(dt)
        self.static_target = self.default.copy()
        self.lpf_state = self.default.copy()
        self.slew_state = self.default.copy()

    def reset(self, static_target: np.ndarray) -> None:
        self.static_target = np.asarray(static_target, dtype=np.float64).copy()
        self.lpf_state = self.static_target.copy()
        self.slew_state = self.static_target.copy()

    def step(self, raw_target: np.ndarray, elapsed_s: float) -> dict[str, np.ndarray | float]:
        raw_target = np.asarray(raw_target, dtype=np.float64)
        safety_alpha = float(
            np.clip(elapsed_s / self.safety_ramp_seconds, 0.0, 1.0)
        )
        safety_target = self.default + safety_alpha * (raw_target - self.default)
        safety_target = np.clip(
            safety_target,
            self.default - self.safety_max_dev,
            self.default + self.safety_max_dev,
        )

        u = float(
            np.clip(elapsed_s / self.supported_ramp_seconds, 0.0, 1.0)
        )
        supported_alpha = u * u * (3.0 - 2.0 * u)
        bounded = self.default + np.clip(
            safety_target - self.default,
            -self.supported_max_dev,
            self.supported_max_dev,
        )
        post_ramp = self.static_target + supported_alpha * (
            bounded - self.static_target
        )

        self.lpf_state += self.lpf_alpha * (post_ramp - self.lpf_state)
        max_step = self.slew_rate * self.dt
        self.slew_state += np.clip(
            self.lpf_state - self.slew_state, -max_step, max_step
        )
        return {
            "safety_alpha": safety_alpha,
            "supported_alpha": supported_alpha,
            "raw_target": raw_target.copy(),
            "safety_target": safety_target.copy(),
            "post_ramp_target": post_ramp.copy(),
            "post_lpf_target": self.lpf_state.copy(),
            "hal_target": self.slew_state.copy(),
        }


# NOTE on the tokenizer layout the fused g1 ONNX expects (verified
# 2026-05-01 by static analysis of the exported graph + parity test
# against a fresh ``dump_isaaclab_step0`` dump):
#
#     The first 680 elements of ``obs`` are reshaped DIRECTLY to
#     (B, 10, 68) by the ONNX graph (single ``Reshape(-1, 10, 68)`` op
#     after a ``Slice``), then flattened back to (B, 680) for the
#     encoder MLP. This means the ONNX expects per-frame *interleaved*
#     layout::
#
#         [cmd_f0(62) | ori_f0(6) | cmd_f1(62) | ori_f1(6) | ... | cmd_f9(62) | ori_f9(6)]
#
#     i.e. exactly what ``np.concatenate([cmd(10,62), ori(10,6)],
#     axis=-1).reshape(-1)`` produces — which is precisely the layout
#     ``eval_x2_mujoco.build_tokenizer_obs`` (and the live IsaacLab
#     ``encoder_input_full``) emits. NO REARRANGEMENT is needed at the
#     ONNX boundary.
#
# History (kept for posterity): an earlier version of this file had a
# ``_interleaved_to_grouped`` rearrangement based on a misreading of
# ``UniversalTokenWrapper.forward()``. That added rearrangement was
# the entire source of the "PT vs ONNX 3.3 rad delta" parity failures
# (e.g. neutral_walk init=20 falling at 1.64 s under ONNX while PT
# saturated). Removing the rearrangement makes ONNX agree with the
# live module to ~5e-7 rad on identical inputs.


# ---------- Offscreen video recorder ----------
class VideoRecorder:
    """Offscreen MuJoCo render piped to ffmpeg (H.264 mp4).

    Same tracking-camera framing as the interactive viewer. Needs ``ffmpeg``
    on PATH; headless GL (set ``MUJOCO_GL=egl`` if there is no display).
    Recording is optional — nothing else depends on it.
    """

    def __init__(self, path: str, model, track_body_id: int, fps: float,
                 width: int = 960, height: int = 720):
        import shutil
        import subprocess
        if shutil.which("ffmpeg") is None:
            raise SystemExit("--record needs ffmpeg on PATH")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = track_body_id
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 120, -20, 2.5
        self.frames = 0
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
             "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             path],
            stdin=subprocess.PIPE,
        )

    def capture(self, data) -> None:
        self.renderer.update_scene(data, camera=self.cam)
        self.proc.stdin.write(self.renderer.render().tobytes())
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()
        print(f"  [record] wrote {self.frames} frames -> {self.path}", flush=True)


def run_kinematic(args) -> None:
    """Kinematic reference playback (no physics, no policy): pose the robot
    straight from the clip frames (PKL ``dof`` is MJCF qpos convention).
    Viewer by default; with ``--record`` renders one offscreen pass over the
    selected clip(s) to mp4 at the clip fps instead."""
    import joblib
    data = joblib.load(args.motion)
    names = list(data.keys())
    if args.clip:
        if args.clip in names:
            names = [args.clip]
        else:
            names = [k for k in names if args.clip.lower() in k.lower()]
        if not names:
            raise SystemExit(f"--clip '{args.clip}' matched no clips in {args.motion}")
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mjd = mujoco.MjData(model)
    pelvis = model.body("pelvis").id

    def pose(m, f):
        mjd.qpos[0:3] = m["root_trans_offset"][f]
        q = np.asarray(m["root_rot"][f])  # xyzw
        mjd.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        mjd.qpos[7:7 + NUM_DOFS] = np.asarray(m["dof"][f])
        mjd.qvel[:] = 0
        mujoco.mj_forward(model, mjd)

    if args.record:
        rec = VideoRecorder(args.record, model, pelvis,
                            float(data[names[0]]["fps"]))
        for name in names:
            m = data[name]
            n_frames = np.asarray(m["dof"]).shape[0]
            print(f"[kinematic] {name}: {n_frames} frames @ {m['fps']:g} fps",
                  flush=True)
            for f in range(args.init_frame, n_frames):
                pose(m, f)
                rec.capture(mjd)
        rec.close()
        return

    state = {"paused": False, "clip": 0, "frame": float(args.init_frame)}

    def key_cb(keycode):
        import glfw
        if keycode == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif keycode == glfw.KEY_R:
            state["frame"] = float(args.init_frame)
        elif keycode == glfw.KEY_N:
            state["clip"] = (state["clip"] + 1) % len(names)
            state["frame"] = float(args.init_frame)

    print("Kinematic playback: SPACE pause, R restart, N next clip.", flush=True)
    with mujoco.viewer.launch_passive(
        model, mjd, key_callback=key_cb,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 120, -20, 2.5
        while viewer.is_running():
            if state["paused"]:
                viewer.sync()
                time.sleep(0.02)
                continue
            m = data[names[state["clip"]]]
            n_frames = np.asarray(m["dof"]).shape[0]
            pose(m, int(state["frame"]) % n_frames)
            viewer.sync()
            time.sleep(1.0 / (float(m["fps"]) * max(args.speed, 1e-6)))
            state["frame"] += 1


# ---------- ONNX wrapper ----------
class OnnxActor:
    """Thin wrapper that mimics ``UniversalTokenActor.__call__`` signature.

    Accepts ``proprioception (990)`` and ``tokenizer_obs (680)`` numpy arrays
    (single batch). ``tokenizer_obs`` must be in the per-frame interleaved
    layout produced by :func:`eval_x2_mujoco.build_tokenizer_obs`; the ONNX
    graph consumes it directly with no rearrangement. See the module-level
    note above for the layout rationale.
    """

    def __init__(self, onnx_path: str, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        # The fused actor is a small MLP: more threads = pure spin-wait
        # overhead. Uncapped, ORT grabs every core (~10 cores busy-spinning
        # per viewer) which starves the GL render loop and drops the sim
        # below real time on laptops. 2 threads is already memory-bound.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_NUM_THREADS", "2"))
        opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=providers
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Expected exactly 1 input and 1 output on the fused ONNX, "
                f"got {len(inputs)} inputs / {len(outputs)} outputs"
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.input_shape = inputs[0].shape
        self.output_shape = outputs[0].shape
        actual_width = self.input_shape[-1]
        if actual_width != ACTOR_OBS_DIM:
            raise RuntimeError(
                f"ONNX input width {actual_width} != expected {ACTOR_OBS_DIM} "
                f"({TOK_DIM} tokenizer + {PROP_DIM} proprioception). Was this ONNX "
                f"exported from a different model than X2 Ultra g1+g1_dyn?"
            )

    def __call__(self, proprioception: np.ndarray, tokenizer_obs: np.ndarray) -> np.ndarray:
        if tokenizer_obs.shape[-1] != TOK_DIM:
            raise ValueError(
                f"Expected tokenizer width {TOK_DIM}, got {tokenizer_obs.shape[-1]}"
            )
        actor_obs = np.concatenate(
            [tokenizer_obs.astype(np.float32), proprioception.astype(np.float32)]
        ).reshape(1, -1)
        out = self.session.run([self.output_name], {self.input_name: actor_obs})[0]
        return out[0]  # (31,) IL order

    def describe(self) -> str:
        return (
            f"input '{self.input_name}' shape={self.input_shape} -> "
            f"output '{self.output_name}' shape={self.output_shape}"
        )



# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--onnx",
        default=None,
        help=(
            "Path to fused encoder+decoder ONNX (e.g. model_step_002000_g1.onnx). "
            "When omitted, falls back to the SONIC model cache: "
            "$SONIC_X2_MODELS/sonic_policy/x2_sonic_policy.onnx, else "
            "$SONIC_HOME/x2/... (SONIC_HOME defaults to ~/.cache/sonic; "
            "populate via install_scripts/setup_x2.sh). Required if no "
            "cached model exists."
        ),
    )
    parser.add_argument("--motion", required=True, help="Reference motion PKL.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--tuning",
        default="__default__",
        help="Real-deploy tuning preset YAML "
        "(gear_sonic_deploy/configs/real_deploy_tuning/*.yaml). Applies the "
        "robot's per-group PD trim (replacing the sim-only ankle bump), "
        "target-deviation clamps, target LPF, and action clip so the sim "
        "matches the real deployment. Default: bigrun.yaml — the preset "
        "the robot is deployed with. Pass --tuning '' for raw "
        "training-parity gains (parity/eval baselines).",
    )
    parser.add_argument(
        "--init-frame",
        type=int,
        default=0,
        help="Motion frame to RSI-initialize the robot at (default 0).",
    )
    parser.add_argument(
        "--init-default-pose",
        action="store_true",
        help="Initialize joints at the trained default pose while keeping the "
             "selected motion reference. This mirrors the suspended real-robot "
             "startup before policy takeover and exposes initial-pose gaps.",
    )
    parser.add_argument(
        "--init-joint-pos-json",
        default=None,
        metavar="PATH",
        help="Optional JSON object mapping all MuJoCo joint names to initial "
             "positions in radians. Applied after the normal RSI/default-pose "
             "initializer for hardware-entry replay experiments.",
    )
    parser.add_argument(
        "--neutral-reference",
        action="store_true",
        help="Replace the loaded clip with a constant DEFAULT_DOF + identity "
             "root-orientation reference, matching C++ StandStillReference.",
    )
    parser.add_argument(
        "--static-hold-seconds",
        type=float,
        default=0.0,
        help="Hold DEFAULT_DOF under static PD before policy entry, then reset "
             "the proprioception history (mirrors GROUND_LOAD_HOLD -> "
             "SUPPORTED_POLICY).",
    )
    parser.add_argument(
        "--policy-ramp-seconds",
        type=float,
        default=0.0,
        help="Smoothstep blend from the static DEFAULT_DOF target to the "
             "bounded policy target after policy entry.",
    )
    parser.add_argument(
        "--safety-ramp-seconds",
        type=float,
        default=0.0,
        help="Linear default-to-policy safety ramp before the supported "
             "smoothstep (exact supported deploy mode only).",
    )
    parser.add_argument(
        "--safety-max-target-dev",
        type=float,
        default=2.0,
        help="Global |target-default| clamp before the supported envelope "
             "(exact supported deploy mode only).",
    )
    parser.add_argument(
        "--supported-deploy-order",
        action="store_true",
        help="Use the real supported command order: safety ramp/clamp, "
             "supported ramp/envelope, LPF, then slew. The tuning YAML's "
             "max_target_dev groups are interpreted as the supported envelope.",
    )
    parser.add_argument(
        "--supported-tilt-delta-deg",
        type=float,
        default=5.0,
        help="Relative pelvis-tilt return threshold for exact supported mode.",
    )
    parser.add_argument(
        "--supported-abs-tilt-deg",
        type=float,
        default=25.0,
        help="Absolute pelvis-tilt return threshold for exact supported mode.",
    )
    parser.add_argument(
        "--supported-joint-vel-trip",
        type=float,
        default=0.8,
        help="Measured |dq| return threshold for exact supported mode.",
    )
    parser.add_argument(
        "--standstill-trace-csv",
        default=None,
        metavar="OUT.csv",
        help="Write per-policy-tick supported command-chain, ankle tracking, "
             "pelvis, support, contact, gain, and torque diagnostics.",
    )
    parser.add_argument(
        "--target-slew-rate",
        type=float,
        default=0.0,
        help="Maximum published-target rate in rad/s after clamp/ramp/LPF "
             "(0 disables).",
    )
    parser.add_argument(
        "--feedback-action",
        choices=("raw", "applied"),
        default="raw",
        help="Value stored in the next proprioception last_action term. "
             "'applied' inverts the final published target, matching the "
             "corrected supported-policy C++ path.",
    )
    parser.add_argument(
        "--init-roll-deg",
        type=float,
        default=0.0,
        help="Override initial pelvis roll for supported-entry A/B tests.",
    )
    parser.add_argument(
        "--init-pitch-deg",
        type=float,
        default=0.0,
        help="Override initial pelvis pitch for supported-entry A/B tests.",
    )
    parser.add_argument(
        "--gantry-load-fraction",
        type=float,
        default=0.0,
        help="Virtual diagnostic gantry load share in [0,1]. It applies a "
             "tether plus this fraction of body weight; 0 disables.",
    )
    parser.add_argument(
        "--gantry-body",
        choices=("pelvis", "torso_link"),
        default="pelvis",
        help="Body receiving the virtual gantry load (default pelvis). Use "
             "torso_link to model the X2 harness attachment near the upper torso.",
    )
    parser.add_argument(
        "--gantry-attachment-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Gantry attachment point in --gantry-body local coordinates, in "
             "metres. Non-zero offsets generate the corresponding physical "
             "moment about that body's centre of mass.",
    )
    parser.add_argument(
        "--gantry-attitude-scale",
        type=float,
        default=1.0,
        help="Scale for the legacy direct roll/pitch stabilising torque. Set "
             "to 0 for a single-point rope whose moment comes only from the "
             "attachment offset (default 1 preserves legacy pelvis tests).",
    )
    parser.add_argument(
        "--gantry-release-start",
        type=float,
        default=-1.0,
        help="Seconds after policy entry to begin releasing the virtual "
             "gantry (-1 keeps it fixed).",
    )
    parser.add_argument(
        "--gantry-release-seconds",
        type=float,
        default=0.0,
        help="Linear virtual-gantry release duration after release-start.",
    )
    parser.add_argument(
        "--fall-height",
        type=float,
        default=0.4,
        help="Pelvis z below this (m) triggers a reset (default 0.4).",
    )
    parser.add_argument(
        "--fall-tilt-cos",
        type=float,
        default=-0.3,
        help="gravity_body[z] above this triggers a reset (default -0.3 ~ 72 deg tilt).",
    )
    parser.add_argument(
        "--max-episode",
        type=float,
        default=0.0,
        help="If > 0, force-reset after this many simulated seconds *per episode* "
        "(default 0 = no per-episode limit).",
    )
    parser.add_argument(
        "--total-sim-seconds",
        type=float,
        default=0.0,
        help="If > 0 (and --no-viewer), exit once *cumulative* simulated seconds "
        "across all episodes reach this. Use this for a fixed-budget parity "
        "rollout that auto-resets through falls (default 0 = no cumulative cap).",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Headless mode: no MuJoCo viewer, no real-time pacing. Pair with "
        "--total-sim-seconds for a deterministic CI-style parity check.",
    )
    parser.add_argument(
        "--record",
        default=None,
        metavar="OUT.mp4",
        help="Record an offscreen 25 fps video of the rollout (headless "
             "only — combine with --no-viewer). Needs ffmpeg on PATH. "
             "With --kinematic, records at the clip fps instead.",
    )
    parser.add_argument(
        "--kinematic",
        action="store_true",
        help="Kinematic reference playback: pose the robot directly from "
             "the clip frames (no physics, no policy, no ONNX needed). "
             "Viewer, or offscreen mp4 with --record.",
    )
    parser.add_argument(
        "--action-clip",
        type=float,
        default=None,
        help="Standalone action clip (rad, IL units) applied even with "
             "--tuning '' (e.g. frozen-g1core v2: --tuning '' "
             "--action-clip 20 --freeze-wrist).",
    )
    parser.add_argument(
        "--freeze-wrist",
        action="store_true",
        help="Zero wrist actions (deploy-default mirror). REQUIRED for the "
             "frozen-g1core v2 model: its wrist axes drift when free.",
    )
    parser.add_argument(
        "--action-audit-json",
        default=None,
        metavar="OUT.json",
        help="Save raw ONNX actions, driven actions, target offsets, snapshots, "
             "and per-joint statistics to JSON.",
    )
    parser.add_argument(
        "--action-print-every",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Print the complete 31-DOF raw ONNX action vector at this interval "
             "(0 disables periodic printing).",
    )
    parser.add_argument(
        "--first-obs-json",
        default=None,
        metavar="OUT.json",
        help="Save the first tokenizer/proprioception input and robot state "
             "to JSON for deploy-vs-sim observation parity analysis.",
    )
    parser.add_argument(
        "--clip",
        default=None,
        help="Kinematic mode: exact clip key, or substring filter "
             "(default: all clips in the PKL).",
    )
    args = parser.parse_args()

    for name in (
        "static_hold_seconds",
        "policy_ramp_seconds",
        "safety_ramp_seconds",
        "target_slew_rate",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.safety_max_target_dev <= 0.0:
        parser.error("--safety-max-target-dev must be > 0")
    if args.supported_deploy_order:
        if args.safety_ramp_seconds <= 0.0:
            parser.error("--supported-deploy-order requires --safety-ramp-seconds > 0")
        if args.policy_ramp_seconds <= 0.0:
            parser.error("--supported-deploy-order requires --policy-ramp-seconds > 0")
        if args.target_slew_rate <= 0.0:
            parser.error("--supported-deploy-order requires --target-slew-rate > 0")
        if not args.neutral_reference:
            parser.error("--supported-deploy-order requires --neutral-reference")
        if not args.freeze_wrist:
            parser.error("--supported-deploy-order requires --freeze-wrist")
        if args.supported_tilt_delta_deg <= 0.0:
            parser.error("--supported-tilt-delta-deg must be > 0")
        if args.supported_abs_tilt_deg <= 0.0:
            parser.error("--supported-abs-tilt-deg must be > 0")
        if args.supported_joint_vel_trip <= 0.0:
            parser.error("--supported-joint-vel-trip must be > 0")
    if args.standstill_trace_csv and not args.supported_deploy_order:
        parser.error("--standstill-trace-csv requires --supported-deploy-order")
    if not 0.0 <= args.gantry_load_fraction <= 1.0:
        parser.error("--gantry-load-fraction must be in [0,1]")
    if args.gantry_attitude_scale < 0.0:
        parser.error("--gantry-attitude-scale must be >= 0")
    if args.gantry_release_seconds < 0.0:
        parser.error("--gantry-release-seconds must be >= 0")

    init_joint_pos_override = None
    if args.init_joint_pos_json:
        init_path = Path(args.init_joint_pos_json)
        try:
            init_payload = json.loads(init_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"failed to read --init-joint-pos-json: {exc}")
        if not isinstance(init_payload, dict):
            parser.error("--init-joint-pos-json must contain a JSON object")
        missing = [name for name in MUJOCO_JOINT_NAMES if name not in init_payload]
        unknown = sorted(set(init_payload) - set(MUJOCO_JOINT_NAMES))
        if missing or unknown:
            parser.error(
                "--init-joint-pos-json joint mismatch: "
                f"missing={missing}, unknown={unknown}"
            )
        try:
            init_joint_pos_override = np.asarray(
                [float(init_payload[name]) for name in MUJOCO_JOINT_NAMES],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid --init-joint-pos-json value: {exc}")

    if args.kinematic:
        if not args.motion:
            parser.error("--kinematic requires --motion")
        run_kinematic(args)
        return

    if args.onnx is None:
        # Default resolution order (mirrors the stack scripts): explicit
        # --onnx > $SONIC_X2_MODELS > $SONIC_HOME/x2 (~/.cache/sonic/x2).
        cache_root = os.environ.get("SONIC_X2_MODELS") or os.path.join(
            os.environ.get("SONIC_HOME", os.path.expanduser("~/.cache/sonic")),
            "x2",
        )
        candidate = os.path.join(cache_root, "sonic_policy", "x2_sonic_policy.onnx")
        if os.path.isfile(candidate):
            print(f"--onnx omitted; using SONIC model cache: {candidate}", flush=True)
            args.onnx = candidate
        else:
            parser.error(
                "--onnx is required (no cached model at "
                f"{candidate}; run install_scripts/setup_x2.sh or pass --onnx)."
            )

    print(f"Loading ONNX session from {args.onnx} ...", flush=True)
    onnx_actor = OnnxActor(args.onnx)
    print(f"  ONNX: {onnx_actor.describe()}", flush=True)

    # Real-deploy tuning preset: swaps in the robot's effective PD gains and
    # reproduces the deploy safety stack (target clamp + LPF + action clip).
    tuning = None
    kp_run, kd_run = KP, KD
    _lpf_y = {"y": None}  # target LPF state; reset with each episode
    _slew_y = {"y": None}
    if args.tuning == "__default__":
        # bigrun is the preset the robot demos this model with; default
        # to it so sim behavior matches the real robot out of the box.
        default_yaml = (
            Path(__file__).resolve().parents[1]
            / "configs/real_deploy_tuning/bigrun.yaml"
        )
        args.tuning = str(default_yaml) if default_yaml.is_file() else ""
    if args.tuning:
        tuning = load_deploy_tuning(args.tuning)
        kp_run, kd_run = tuning["kp"], tuning["kd"]
        print(
            f"  [tuning] {args.tuning}: PD trim + target clamps + LPF + "
            f"action_clip={tuning['action_clip']:g} (robot-matched)",
            flush=True,
        )
    if args.supported_deploy_order and tuning is None:
        parser.error("--supported-deploy-order requires a tuning YAML")

    supported_shaper = None
    if args.supported_deploy_order:
        supported_shaper = SupportedDeployShaper(
            default=DEFAULT_DOF,
            safety_max_dev=np.full(NUM_DOFS, args.safety_max_target_dev),
            supported_max_dev=tuning["max_target_dev"],
            lpf_alpha=tuning["lpf_alpha"],
            safety_ramp_seconds=args.safety_ramp_seconds,
            supported_ramp_seconds=args.policy_ramp_seconds,
            slew_rate=args.target_slew_rate,
            dt=CONTROL_DT,
        )
        print(
            "  [supported exact] safety ramp "
            f"{args.safety_ramp_seconds:g}s -> supported ramp "
            f"{args.policy_ramp_seconds:g}s -> LPF -> "
            f"slew {args.target_slew_rate:g}rad/s; "
            f"ankle Kp/Kd={kp_run[4]:.6f}/{kd_run[4]:.6f}; "
            f"last_action={args.feedback_action}",
            flush=True,
        )

    wrist_mj_idx = ([i for i, n in enumerate(MUJOCO_JOINT_NAMES)
                     if "wrist" in n] if args.freeze_wrist else [])
    if args.freeze_wrist:
        print(f"  [wrist] freezing {len(wrist_mj_idx)} wrist joints "
              f"(deploy-default mirror)", flush=True)

    print(f"Loading motion from {args.motion} ...", flush=True)
    motion_data = load_motion_data(args.motion)
    if args.neutral_reference:
        motion = motion_data[next(iter(motion_data))]
        frame_count = np.asarray(motion["dof"]).shape[0]
        motion["dof"] = np.repeat(DEFAULT_DOF[None, :], frame_count, axis=0)
        motion["root_rot"] = np.repeat(
            np.asarray([[0.0, 0.0, 0.0, 1.0]]), frame_count, axis=0)
        root_pos = np.asarray(motion["root_trans_offset"])[0].copy()
        motion["root_trans_offset"] = np.repeat(
            root_pos[None, :], frame_count, axis=0)
        print("  [reference] neutral DEFAULT_DOF + identity root", flush=True)
    total_frames = get_total_frames(motion_data)
    motion_fps = get_motion_fps(motion_data)
    print(
        f"  {total_frames} frames @ {motion_fps} fps = {total_frames / motion_fps:.1f}s",
        flush=True,
    )

    print("Loading MuJoCo model ...", flush=True)
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = SIM_DT

    pelvis_id = mj_model.body("pelvis").id
    robot_mass = float(mj_model.body_subtreemass[pelvis_id])
    gantry_body_id = mj_model.body(args.gantry_body).id
    gantry_attachment_offset = np.asarray(
        args.gantry_attachment_offset, dtype=np.float64
    )
    legacy_pelvis_gantry = (
        args.gantry_body == "pelvis"
        and np.array_equal(gantry_attachment_offset, np.zeros(3))
    )
    print(
        "  [gantry] "
        f"body={args.gantry_body} "
        f"offset={gantry_attachment_offset.tolist()}m "
        f"attitude_scale={args.gantry_attitude_scale:g}",
        flush=True,
    )

    floor_geom_id = mj_model.geom("floor").id
    foot_geom_groups: dict[str, dict[str, list[int]]] = {}
    for side in ("left", "right"):
        foot_body_id = mj_model.body(f"{side}_ankle_roll_link").id
        sphere_ids = [
            geom_id
            for geom_id in range(mj_model.ngeom)
            if int(mj_model.geom_bodyid[geom_id]) == foot_body_id
            and int(mj_model.geom_type[geom_id])
            == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        ]
        foot_geom_groups[side] = {
            "all": sphere_ids,
            "heel": [
                geom_id for geom_id in sphere_ids
                if float(mj_model.geom_pos[geom_id, 0]) <= -0.04
            ],
            "toe": [
                geom_id for geom_id in sphere_ids
                if float(mj_model.geom_pos[geom_id, 0]) >= 0.10
            ],
        }

    trace_stream = None
    trace_writer = None
    if args.standstill_trace_csv:
        trace_path = Path(args.standstill_trace_csv)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_stream = trace_path.open("w", encoding="utf-8", newline="")
        common_columns = [
            "episode", "policy_t", "gantry_fraction", "gantry_body",
            "gantry_attach_x", "gantry_attach_y", "gantry_attach_z",
            "gantry_force_x", "gantry_force_y", "gantry_force_z",
            "gantry_torque_x", "gantry_torque_y", "gantry_torque_z",
            "safety_alpha", "supported_alpha",
            "pelvis_z", "pelvis_roll_deg", "pelvis_pitch_deg",
            "pelvis_tilt_deg", "max_abs_dq", "max_abs_dq_joint",
            "left_contact_count", "right_contact_count",
            "left_contact_normal_n", "right_contact_normal_n",
            "foot_load_fraction",
            "left_toe_clearance", "left_heel_clearance",
            "right_toe_clearance", "right_heel_clearance",
        ]
        trace_joint_specs = (
            ("left", 4),
            ("right", 10),
            ("waist_pitch", 13),
            ("waist_roll", 14),
        )
        joint_columns = []
        for prefix, _ in trace_joint_specs:
            joint_columns.extend(
                f"{prefix}_{name}" for name in (
                    "policy_action", "driven_action", "raw_target",
                    "safety_target", "post_ramp_target", "post_lpf_target",
                    "hal_target", "measured_q", "measured_dq",
                    "tracking_error", "kp", "kd", "commanded_pd_torque",
                    "poststep_pd_estimate",
                )
            )
        trace_writer = csv.writer(trace_stream)
        trace_writer.writerow(common_columns + joint_columns)
        print(f"  [standstill trace] {trace_path}", flush=True)

    recorder = None
    if args.record:
        if not args.no_viewer:
            parser.error("--record requires --no-viewer (offscreen render)")
        recorder = VideoRecorder(args.record, mj_model, pelvis_id, 25.0)
        print(f"Recording -> {args.record} @ 25 fps", flush=True)

    init_frame = int(args.init_frame)
    init_motion_state = compute_motion_state(motion_data, init_frame, motion_fps)
    init_root_z = float(init_motion_state["root_pos_w"][2])
    print(
        f"  [RSI] Initializing from motion frame {init_frame} "
        f"(t={init_frame / motion_fps:.3f}s)",
        flush=True,
    )

    prop_buf = ProprioceptionBuffer()
    last_action_mj = np.zeros(NUM_DOFS, dtype=np.float32)
    sim_time = float(init_frame) / motion_fps
    step_count = 0
    episode_count = 0
    episode_start_step = 0
    paused = False
    policy_started = False
    policy_entry_tilt_deg = 0.0
    gantry_anchor_pos = np.zeros(3, dtype=np.float64)
    gantry_anchor_roll_pitch = np.zeros(2, dtype=np.float64)
    gantry_attachment_pos = np.zeros(3, dtype=np.float64)
    gantry_applied_force = np.zeros(3, dtype=np.float64)
    gantry_applied_torque = np.zeros(3, dtype=np.float64)

    def _roll_pitch_tilt(quat_wxyz):
        w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
        roll = math.atan2(2.0 * (w * x + y * z),
                          1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
        return roll, pitch, math.hypot(roll, pitch)

    def _gantry_fraction(episode_seconds):
        fraction = args.gantry_load_fraction
        if fraction <= 0.0 or not policy_started or args.gantry_release_start < 0.0:
            return fraction
        policy_elapsed = max(0.0, episode_seconds - args.static_hold_seconds)
        if policy_elapsed <= args.gantry_release_start:
            return fraction
        if args.gantry_release_seconds <= 0.0:
            return 0.0
        release = np.clip(
            (policy_elapsed - args.gantry_release_start)
            / args.gantry_release_seconds,
            0.0,
            1.0,
        )
        return fraction * (1.0 - float(release))

    def _gantry_body_state():
        if legacy_pelvis_gantry:
            return (
                np.asarray(mj_data.qpos[0:3], dtype=np.float64).copy(),
                np.asarray(mj_data.qvel[0:3], dtype=np.float64).copy(),
                np.asarray(mj_data.qvel[3:6], dtype=np.float64).copy(),
                np.asarray(mj_data.qpos[3:7], dtype=np.float64).copy(),
                np.zeros(3, dtype=np.float64),
            )

        body_pos = np.asarray(mj_data.xpos[gantry_body_id], dtype=np.float64)
        body_rot = np.asarray(
            mj_data.xmat[gantry_body_id], dtype=np.float64
        ).reshape(3, 3)
        offset_world = body_rot @ gantry_attachment_offset
        attachment_pos = body_pos + offset_world
        spatial_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            mj_model,
            mj_data,
            mujoco.mjtObj.mjOBJ_BODY,
            gantry_body_id,
            spatial_velocity,
            0,
        )
        angular_velocity = spatial_velocity[0:3]
        attachment_velocity = (
            spatial_velocity[3:6]
            + np.cross(angular_velocity, offset_world)
        )
        body_com = np.asarray(
            mj_data.xipos[gantry_body_id], dtype=np.float64
        ).copy()
        return (
            attachment_pos.copy(),
            attachment_velocity.copy(),
            angular_velocity.copy(),
            np.asarray(mj_data.xquat[gantry_body_id], dtype=np.float64).copy(),
            attachment_pos - body_com,
        )

    def _apply_virtual_gantry(episode_seconds):
        mj_data.xfrc_applied[:] = 0.0
        gantry_applied_force[:] = 0.0
        gantry_applied_torque[:] = 0.0
        fraction = _gantry_fraction(episode_seconds)
        pos, lin_vel, ang_vel, quat, moment_arm = _gantry_body_state()
        gantry_attachment_pos[:] = pos
        if fraction <= 0.0:
            return 0.0

        roll, pitch, _ = _roll_pitch_tilt(quat)

        # A deliberately simple diagnostic harness: unload body weight and
        # tether one body attachment point. Every term scales to zero during
        # release, so the final interval is genuine policy-only physics.
        # The load-bearing test condition is closer to a short, taut gantry
        # than a compliant bungee. Keep this stiff enough that static PD does
        # not tip the unsupported floating base before policy entry; the
        # release schedule still scales every term continuously to zero.
        k_xy, d_xy = 1200.0, 120.0
        k_z, d_z = 3000.0, 220.0
        k_rp, d_rp = 500.0, 45.0
        force = np.zeros(3, dtype=np.float64)
        force[0:2] = fraction * (
            k_xy * (gantry_anchor_pos[0:2] - pos[0:2])
            - d_xy * lin_vel[0:2]
        )
        force[2] = fraction * (
            robot_mass * 9.81
            + k_z * (gantry_anchor_pos[2] - pos[2])
            - d_z * lin_vel[2]
        )
        force[2] = float(np.clip(force[2], 0.0, 2.0 * robot_mass * 9.81))
        torque = np.cross(moment_arm, force)
        attitude_torque = np.zeros(3, dtype=np.float64)
        attitude_torque[0] = args.gantry_attitude_scale * fraction * (
            k_rp * (gantry_anchor_roll_pitch[0] - roll) - d_rp * ang_vel[0]
        )
        attitude_torque[1] = args.gantry_attitude_scale * fraction * (
            k_rp * (gantry_anchor_roll_pitch[1] - pitch) - d_rp * ang_vel[1]
        )
        torque += attitude_torque
        gantry_applied_force[:] = force
        gantry_applied_torque[:] = torque
        mj_data.xfrc_applied[gantry_body_id, 0:3] = force
        mj_data.xfrc_applied[gantry_body_id, 3:6] = torque
        return fraction

    joint_names_il = [MUJOCO_JOINT_NAMES[i] for i in IL_TO_MJ_DOF]
    action_audit = {"raw": [], "driven": [], "target_offset": []}
    first_obs_saved = False

    def _audit_stats(samples):
        values = np.asarray(samples, dtype=np.float64)
        jumps = np.abs(np.diff(values, axis=0))
        return {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "abs_max": np.abs(values).max(axis=0).tolist(),
            "max_frame_jump": (
                jumps.max(axis=0).tolist() if len(jumps) else [0.0] * NUM_DOFS
            ),
        }

    def _record_action_audit(raw_il, driven_mj, target_pos_mj):
        raw = np.asarray(raw_il, dtype=np.float32).copy()
        driven = np.asarray(driven_mj[IL_TO_MJ_DOF], dtype=np.float32).copy()
        target_offset = np.asarray(
            target_pos_mj[IL_TO_MJ_DOF] - DEFAULT_DOF[IL_TO_MJ_DOF],
            dtype=np.float32,
        )
        action_audit["raw"].append(raw)
        action_audit["driven"].append(driven)
        action_audit["target_offset"].append(target_offset)

        interval_steps = (
            max(1, int(round(args.action_print_every / CONTROL_DT)))
            if args.action_print_every > 0 else 0
        )
        sample_idx = len(action_audit["raw"]) - 1
        if interval_steps and sample_idx % interval_steps == 0:
            values = ", ".join(
                f"{name.replace('_joint', '')}={value:+.3f}"
                for name, value in zip(joint_names_il, raw)
            )
            print(
                f"[action raw_il] step={sample_idx} t={sample_idx * CONTROL_DT:.2f}s "
                f"{values}",
                flush=True,
            )

    def _finish_action_audit():
        if not action_audit["raw"]:
            return
        raw_stats = _audit_stats(action_audit["raw"])
        order = np.argsort(np.asarray(raw_stats["abs_max"]))[::-1]
        print("\n=== Raw ONNX action audit (IL order) ===", flush=True)
        print("joint                                min      max     mean  abs_max     jump")
        for idx in order:
            print(
                f"{joint_names_il[idx]:<34} "
                f"{raw_stats['min'][idx]:+8.3f} "
                f"{raw_stats['max'][idx]:+8.3f} "
                f"{raw_stats['mean'][idx]:+8.3f} "
                f"{raw_stats['abs_max'][idx]:8.3f} "
                f"{raw_stats['max_frame_jump'][idx]:8.3f}",
                flush=True,
            )

        if not args.action_audit_json:
            return
        out_path = Path(args.action_audit_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw = action_audit["raw"]
        one_second_idx = min(int(round(1.0 / CONTROL_DT)), len(raw) - 1)
        payload = {
            "sample_hz": 1.0 / CONTROL_DT,
            "sample_count": len(raw),
            "joint_order": "IsaacLab",
            "joint_names": joint_names_il,
            "snapshots": {
                "first_raw": raw[0].tolist(),
                "one_second_raw": raw[one_second_idx].tolist(),
            },
            "raw": raw_stats,
            "driven": _audit_stats(action_audit["driven"]),
            "target_offset_rad": _audit_stats(action_audit["target_offset"]),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[action audit] wrote {out_path}", flush=True)

    def _apply_init_state():
        _lpf_y["y"] = None
        _slew_y["y"] = None
        s = init_motion_state
        mj_data.qpos[0] = 0.0
        mj_data.qpos[1] = 0.0
        mj_data.qpos[2] = float(s["root_pos_w"][2])
        if args.init_roll_deg != 0.0 or args.init_pitch_deg != 0.0:
            roll = math.radians(args.init_roll_deg)
            pitch = math.radians(args.init_pitch_deg)
            cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
            cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
            # R = Rz(0) * Ry(pitch) * Rx(roll), returned as wxyz.
            mj_data.qpos[3:7] = [cp * cr, cp * sr, sp * cr, -sp * sr]
        else:
            mj_data.qpos[3:7] = s["root_quat_w_wxyz"]
        if args.init_default_pose:
            mj_data.qpos[7 : 7 + NUM_DOFS] = DEFAULT_DOF
            mj_data.qvel[:] = 0.0
        else:
            mj_data.qpos[7 : 7 + NUM_DOFS] = s["joint_pos_mj"]
            mj_data.qvel[0:3] = s["root_lin_vel_w"]
            mj_data.qvel[3:6] = quat_rotate_inverse(
                s["root_quat_w_wxyz"], s["root_ang_vel_w"]
            )
            mj_data.qvel[6 : 6 + NUM_DOFS] = s["joint_vel_mj"]
        if init_joint_pos_override is not None:
            mj_data.qpos[7 : 7 + NUM_DOFS] = init_joint_pos_override
            mj_data.qvel[:] = 0.0
        mj_data.xfrc_applied[:] = 0
        mujoco.mj_forward(mj_model, mj_data)
        pos, _, _, quat, _ = _gantry_body_state()
        gantry_anchor_pos[:] = pos
        gantry_attachment_pos[:] = pos
        gantry_applied_force[:] = 0.0
        gantry_applied_torque[:] = 0.0
        roll, pitch, _ = _roll_pitch_tilt(quat)
        gantry_anchor_roll_pitch[:] = [roll, pitch]

    _apply_init_state()

    # Per-episode tracking-error accumulator vs the reference clip (joint
    # MAE in rad, max joint error, pelvis-z MAE in m). Reset each episode;
    # printed at every episode end.
    _ref_clip = motion_data[list(motion_data.keys())[0]]
    _track = {"jmae": 0.0, "jmax": 0.0, "zmae": 0.0, "n": 0}

    def _track_reset():
        _track.update(jmae=0.0, jmax=0.0, zmae=0.0, n=0)

    def _track_summary() -> str:
        n = max(_track["n"], 1)
        return (f"joint MAE {_track['jmae'] / n:.4f} rad "
                f"(max {_track['jmax']:.3f}) | pelvis-z MAE "
                f"{_track['zmae'] / n:.3f} m")

    # Tracks the reference index so we can catch the loop wrap. Sonic is a tracking
    # policy: it can neither hold a frozen frame nor chase a teleport. Letting the
    # reference index wrap on its own snaps it back to frame 0 -- metres behind the
    # robot on locomotion clips -- and the robot collapses. Wrapping must be a full
    # reset so sim state and reference move together.
    prev_motion_frame = -1

    def reset_state(reason: str = "") -> None:
        nonlocal sim_time, last_action_mj, episode_count, episode_start_step
        nonlocal prev_motion_frame, policy_started, policy_entry_tilt_deg
        if _track["n"]:
            print(f"  [tracking] {_track_summary()}", flush=True)
        _track_reset()
        prev_motion_frame = -1
        sim_time = float(init_frame) / motion_fps
        last_action_mj[:] = 0
        prop_buf.reset()
        policy_started = False
        policy_entry_tilt_deg = 0.0
        _apply_init_state()
        episode_count += 1
        episode_start_step = step_count
        tag = f" ({reason})" if reason else ""
        print(f"\n[reset]{tag} starting episode {episode_count}", flush=True)

    # Per-step body that's identical in headless and viewer paths.
    def step_once() -> str | None:
        """Run one control tick. Returns reset reason if the episode ended."""
        nonlocal sim_time, step_count, last_action_mj, prev_motion_frame
        nonlocal first_obs_saved, policy_started, policy_entry_tilt_deg

        if args.neutral_reference:
            # StandStill is an infinite constant reference. Keeping the source
            # PKL's finite frame count here used to terminate/reset a neutral
            # rollout when that otherwise-unused clip wrapped (9.6 s for the
            # retained idle clip), invalidating long supported simulations.
            motion_frame = 0
            motion_time = 0.0
        else:
            motion_time = sim_time * args.speed
            motion_frame = int(motion_time * motion_fps) % total_frames
            # Clip looped: end the episode instead of teleporting the reference (see above).
            if 0 <= prev_motion_frame and motion_frame < prev_motion_frame:
                return "motion_end"
            prev_motion_frame = motion_frame
            motion_time = motion_frame / motion_fps

        qpos_j = mj_data.qpos[7 : 7 + NUM_DOFS].copy()
        qvel_j = mj_data.qvel[6 : 6 + NUM_DOFS].copy()
        base_quat = mj_data.qpos[3:7].copy()
        base_angvel = mj_data.qvel[3:6].copy()

        episode_seconds = (step_count - episode_start_step) * CONTROL_DT
        in_static_hold = episode_seconds < args.static_hold_seconds
        if not in_static_hold and not policy_started:
            prop_buf.reset()
            last_action_mj[:] = 0.0
            _lpf_y["y"] = None
            _slew_y["y"] = DEFAULT_DOF.copy()
            if supported_shaper is not None:
                supported_shaper.reset(DEFAULT_DOF)
            policy_started = True
            entry_roll, entry_pitch, entry_tilt = _roll_pitch_tilt(base_quat)
            policy_entry_tilt_deg = math.degrees(entry_tilt)
            print(
                f"  [policy entry] t={episode_seconds:.2f}s "
                f"roll={math.degrees(entry_roll):+.2f}deg "
                f"pitch={math.degrees(entry_pitch):+.2f}deg "
                f"tilt={math.degrees(entry_tilt):.2f}deg",
                flush=True,
            )

        dof_pos_il = qpos_j[IL_TO_MJ_DOF]
        dof_vel_il = qvel_j[IL_TO_MJ_DOF]
        action_il = last_action_mj[IL_TO_MJ_DOF]

        gravity = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))
        dof_pos_rel_il = dof_pos_il - DEFAULT_DOF[IL_TO_MJ_DOF]

        if in_static_hold:
            action_il_onnx = np.zeros(NUM_DOFS, dtype=np.float32)
            tokenizer_obs = None
            proprioception = None
        else:
            prop_buf.append(
                gravity, base_angvel, dof_pos_rel_il, dof_vel_il, action_il
            )
            proprioception = prop_buf.get_flat()
            tokenizer_obs = build_tokenizer_obs(
                motion_data, motion_time, base_quat, motion_fps
            )
            action_il_onnx = onnx_actor(proprioception, tokenizer_obs)

        if (not in_static_hold and args.first_obs_json and not first_obs_saved):
            first_obs_path = Path(args.first_obs_json)
            first_obs_path.parent.mkdir(parents=True, exist_ok=True)
            first_obs_path.write_text(json.dumps({
                "policy_time": motion_time,
                "tokenizer_obs": np.asarray(tokenizer_obs).tolist(),
                "proprioception": np.asarray(proprioception).tolist(),
                "action_il": np.asarray(action_il_onnx).tolist(),
                "joint_pos_mj": np.asarray(qpos_j).tolist(),
                "joint_vel_mj": np.asarray(qvel_j).tolist(),
                "base_quat_wxyz": np.asarray(base_quat).tolist(),
                "base_ang_vel": np.asarray(base_angvel).tolist(),
            }, indent=2), encoding="utf-8")
            first_obs_saved = True
            print(f"[first obs] wrote {first_obs_path}", flush=True)


        action_il_drive = action_il_onnx

        action_mj = action_il_drive[MJ_TO_IL_DOF]
        if tuning is not None:
            action_mj = np.clip(
                action_mj, -tuning["action_clip"], tuning["action_clip"]
            )
        if args.action_clip is not None:
            action_mj = np.clip(action_mj, -args.action_clip, args.action_clip)
        if wrist_mj_idx:
            action_mj[wrist_mj_idx] = 0.0
        target_pos = DEFAULT_DOF + action_mj * ACTION_SCALE
        if in_static_hold:
            target_pos = DEFAULT_DOF.copy()
        stages = None
        if tuning is not None and not args.supported_deploy_order:
            # Deploy safety stack: clamp the target around the default pose,
            # then low-pass filter it (both per joint group).
            target_pos = np.clip(
                target_pos,
                DEFAULT_DOF - tuning["max_target_dev"],
                DEFAULT_DOF + tuning["max_target_dev"],
            )
            if _lpf_y["y"] is None:
                _lpf_y["y"] = target_pos.copy()
            else:
                _lpf_y["y"] += tuning["lpf_alpha"] * (target_pos - _lpf_y["y"])
            target_pos = _lpf_y["y"].copy()

        if not in_static_hold and args.supported_deploy_order:
            policy_elapsed = max(0.0, episode_seconds - args.static_hold_seconds)
            stages = supported_shaper.step(target_pos, policy_elapsed)
            target_pos = stages["hal_target"]
        elif not in_static_hold and args.policy_ramp_seconds > 0.0:
            policy_elapsed = max(0.0, episode_seconds - args.static_hold_seconds)
            u = np.clip(policy_elapsed / args.policy_ramp_seconds, 0.0, 1.0)
            alpha = u * u * (3.0 - 2.0 * u)
            target_pos = DEFAULT_DOF + alpha * (target_pos - DEFAULT_DOF)

        if (
            not in_static_hold
            and not args.supported_deploy_order
            and args.target_slew_rate > 0.0
        ):
            if _slew_y["y"] is None:
                _slew_y["y"] = DEFAULT_DOF.copy()
            max_step = args.target_slew_rate * CONTROL_DT
            _slew_y["y"] += np.clip(
                target_pos - _slew_y["y"], -max_step, max_step
            )
            target_pos = _slew_y["y"].copy()

        if in_static_hold:
            last_action_mj = np.zeros(NUM_DOFS, dtype=np.float32)
        elif args.feedback_action == "applied":
            last_action_mj = (
                (target_pos - DEFAULT_DOF) / ACTION_SCALE
            ).astype(np.float32)
        else:
            last_action_mj = action_mj.astype(np.float32).copy()

        _record_action_audit(action_il_onnx, action_mj, target_pos)

        gantry_fraction = 0.0
        torque = np.zeros(NUM_DOFS, dtype=np.float64)
        for _ in range(DECIMATION):
            gantry_fraction = _apply_virtual_gantry(episode_seconds)
            torque = (
                kp_run * (target_pos - mj_data.qpos[7 : 7 + NUM_DOFS])
                - kd_run * mj_data.qvel[6 : 6 + NUM_DOFS]
            )
            for j in range(NUM_DOFS):
                mj_data.ctrl[JOINT_TO_ACTUATOR[j]] = torque[j]
            mujoco.mj_step(mj_model, mj_data)

        sim_time += CONTROL_DT
        step_count += 1

        current_q = mj_data.qpos[7 : 7 + NUM_DOFS].copy()
        current_dq = mj_data.qvel[6 : 6 + NUM_DOFS].copy()
        current_roll, current_pitch, current_tilt = _roll_pitch_tilt(
            mj_data.qpos[3:7]
        )
        current_tilt_deg = math.degrees(current_tilt)
        max_abs_dq_index = int(np.argmax(np.abs(current_dq)))
        max_abs_dq = float(np.abs(current_dq[max_abs_dq_index]))

        def _foot_clearance(side: str, part: str) -> float:
            ids = foot_geom_groups[side][part]
            return min(
                float(mj_data.geom_xpos[geom_id, 2] - mj_model.geom_size[geom_id, 0])
                for geom_id in ids
            )

        def _foot_contact_state(side: str) -> tuple[int, float]:
            foot_ids = set(foot_geom_groups[side]["all"])
            count = 0
            normal_force = 0.0
            contact_force = np.zeros(6, dtype=np.float64)
            for contact_id in range(mj_data.ncon):
                contact = mj_data.contact[contact_id]
                pair = {int(contact.geom1), int(contact.geom2)}
                if floor_geom_id in pair and pair.intersection(foot_ids):
                    count += 1
                    mujoco.mj_contactForce(
                        mj_model, mj_data, contact_id, contact_force
                    )
                    normal_force += abs(float(contact_force[0]))
            return count, normal_force

        if trace_writer is not None and not in_static_hold:
            policy_elapsed = max(0.0, episode_seconds - args.static_hold_seconds)
            left_contact_count, left_contact_normal = _foot_contact_state("left")
            right_contact_count, right_contact_normal = _foot_contact_state("right")
            row = [
                episode_count,
                policy_elapsed,
                gantry_fraction,
                args.gantry_body,
                *gantry_attachment_pos.tolist(),
                *gantry_applied_force.tolist(),
                *gantry_applied_torque.tolist(),
                stages["safety_alpha"],
                stages["supported_alpha"],
                float(mj_data.qpos[2]),
                math.degrees(current_roll),
                math.degrees(current_pitch),
                current_tilt_deg,
                max_abs_dq,
                MUJOCO_JOINT_NAMES[max_abs_dq_index],
                left_contact_count,
                right_contact_count,
                left_contact_normal,
                right_contact_normal,
                (left_contact_normal + right_contact_normal)
                / (robot_mass * 9.81),
                _foot_clearance("left", "toe"),
                _foot_clearance("left", "heel"),
                _foot_clearance("right", "toe"),
                _foot_clearance("right", "heel"),
            ]
            for _, mj_index in trace_joint_specs:
                il_index = MJ_TO_IL_DOF[mj_index]
                tracking = float(target_pos[mj_index] - current_q[mj_index])
                row.extend(
                    [
                        float(action_il_onnx[il_index]),
                        float(action_mj[mj_index]),
                        float(stages["raw_target"][mj_index]),
                        float(stages["safety_target"][mj_index]),
                        float(stages["post_ramp_target"][mj_index]),
                        float(stages["post_lpf_target"][mj_index]),
                        float(target_pos[mj_index]),
                        float(current_q[mj_index]),
                        float(current_dq[mj_index]),
                        tracking,
                        float(kp_run[mj_index]),
                        float(kd_run[mj_index]),
                        float(torque[mj_index]),
                        float(
                            kp_run[mj_index] * tracking
                            - kd_run[mj_index] * current_dq[mj_index]
                        ),
                    ]
                )
            trace_writer.writerow(row)

        pelvis_z = float(mj_data.qpos[2])
        _err = np.abs(mj_data.qpos[7:7 + NUM_DOFS]
                      - np.asarray(_ref_clip["dof"][motion_frame]))
        _track["jmae"] += float(_err.mean())
        _track["jmax"] = max(_track["jmax"], float(_err.max()))
        _track["zmae"] += abs(pelvis_z
                              - float(_ref_clip["root_trans_offset"][motion_frame][2]))
        _track["n"] += 1
        current_gravity = quat_rotate_inverse(
            mj_data.qpos[3:7], np.array([0.0, 0.0, -1.0])
        )
        grav_z = float(current_gravity[2])
        if not in_static_hold and args.supported_deploy_order:
            if (
                current_tilt_deg >= args.supported_abs_tilt_deg
                or current_tilt_deg
                >= policy_entry_tilt_deg + args.supported_tilt_delta_deg
            ):
                return (
                    f"supported tilt gate: {current_tilt_deg:.2f}deg "
                    f"(entry {policy_entry_tilt_deg:.2f}deg)"
                )
            if max_abs_dq >= args.supported_joint_vel_trip:
                return (
                    "supported velocity gate: "
                    f"{MUJOCO_JOINT_NAMES[max_abs_dq_index]} "
                    f"{max_abs_dq:.3f}rad/s "
                    f">= {args.supported_joint_vel_trip:.3f}rad/s"
                )
        if pelvis_z < args.fall_height:
            return f"pelvis_z={pelvis_z:.3f} < {args.fall_height:.2f}"
        if grav_z > args.fall_tilt_cos:
            tilt_deg = int(np.rad2deg(np.arccos(np.clip(-grav_z, -1, 1))))
            return (
                f"gravity_body[z]={grav_z:+.2f} > {args.fall_tilt_cos:.2f} "
                f"(tilt {tilt_deg} deg)"
            )
        if args.max_episode > 0 and episode_seconds >= args.max_episode:
            return f"reached --max-episode={args.max_episode:.1f}s"
        if step_count % 250 == 0:
            phase = "static" if in_static_hold else "policy"
            tilt_deg = float(np.rad2deg(np.arccos(np.clip(-grav_z, -1, 1))))
            print(
                f"[ep {episode_count}] step={step_count} sim={sim_time:.2f}s "
                f"frame={motion_frame}/{total_frames} phase={phase} "
                f"h={pelvis_z:.3f}m tilt={tilt_deg:.2f}deg "
                f"gantry={gantry_fraction:.2f}",
                flush=True,
            )
        return None

    print("\n=== X2 MuJoCo Eval (ONNX) ===", flush=True)
    if args.init_default_pose:
        print(
            f"Robot joints initialized at DEFAULT_DOF; reference starts at "
            f"motion frame {init_frame}.",
            flush=True,
        )
    else:
        print(f"Robot RSI-initialized from motion frame {init_frame}.", flush=True)
    print(
        f"Auto-reset triggers: pelvis_z < {args.fall_height:.2f} m, "
        f"or gravity_body[z] > {args.fall_tilt_cos:.2f}.",
        flush=True,
    )
    if args.max_episode > 0:
        print(f"Max episode length: {args.max_episode:.1f} s.", flush=True)
    if not args.no_viewer:
        print("Press SPACE pause, R reset, V toggle camera.\n", flush=True)
    else:
        print("Headless mode (no viewer).\n", flush=True)

    # Headless exit semantics:
    #   --total-sim-seconds > 0  -> keep cycling resets until cumulative sim
    #                                time hits the cap (good for parity budgets)
    #   --max-episode > 0 only   -> exit after the first episode terminates
    #   neither set              -> run forever (Ctrl-C to stop)
    headless_exit_after_one_episode = (
        args.no_viewer and args.max_episode > 0 and args.total_sim_seconds <= 0
    )
    exit_requested = False
    cumulative_sim_seconds = 0.0

    if args.no_viewer:
        # Tight loop with no real-time pacing or viewer sync.
        while not exit_requested:
            reason = step_once()
            # 50 Hz control -> capture every 2nd tick = 25 fps video.
            if recorder is not None and reason is None and step_count % 2 == 0:
                recorder.capture(mj_data)
            cumulative_sim_seconds += CONTROL_DT
            if (
                args.total_sim_seconds > 0
                and cumulative_sim_seconds >= args.total_sim_seconds
            ):
                print(
                    f"  [end] cumulative sim time {cumulative_sim_seconds:.2f}s "
                    f">= --total-sim-seconds={args.total_sim_seconds:.1f}s, exiting.",
                    flush=True,
                )
                exit_requested = True
                continue
            if reason is not None:
                print(
                    f"  [end] ep={episode_count} ran "
                    f"{(step_count - episode_start_step) * CONTROL_DT:.2f}s, "
                    f"reason: {reason} | {_track_summary()}",
                    flush=True,
                )
                _track_reset()
                if headless_exit_after_one_episode:
                    exit_requested = True
                else:
                    reset_state(reason)
        if recorder is not None:
            recorder.close()
        _finish_action_audit()
        if trace_stream is not None:
            trace_stream.close()
            print(f"[standstill trace] closed {args.standstill_trace_csv}", flush=True)
    else:

        def key_callback(keycode):
            nonlocal paused
            import glfw

            if keycode == glfw.KEY_SPACE:
                paused = not paused
                print("Paused" if paused else "Resumed", flush=True)
            elif keycode == glfw.KEY_R:
                reset_state("manual")
            elif keycode == glfw.KEY_V:
                if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                else:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = pelvis_id

        with mujoco.viewer.launch_passive(
            mj_model,
            mj_data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20
            viewer.cam.distance = 3.0
            viewer.cam.lookat[:] = [0.0, 0.0, init_root_z]
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = pelvis_id

            wall_start = time.time() - sim_time

            # viewer.sync() blocks on the display's vsync (~17 ms at 60 Hz),
            # so syncing every 50 Hz control step caps the sim below real
            # time. Render every Nth step instead: physics stays 50 Hz,
            # display runs at 25 Hz (override with VIEWER_RENDER_STRIDE=1
            # on machines with fast/uncomposited GL).
            render_stride = max(1, int(os.environ.get("VIEWER_RENDER_STRIDE", "2")))

            while viewer.is_running():
                if paused:
                    viewer.sync()
                    time.sleep(0.02)
                    continue

                reason = step_once()
                if step_count % render_stride == 0:
                    viewer.sync()

                wall_elapsed = time.time() - wall_start
                if sim_time > wall_elapsed:
                    time.sleep(sim_time - wall_elapsed)
                elif wall_elapsed - sim_time > 0.5:
                    # Fell >0.5 s behind (slow GL, window drag, CPU spike):
                    # rebase so we resume real-time pacing instead of
                    # chasing the deficit in permanent fast-forward.
                    wall_start = time.time() - sim_time

                if reason is not None:
                    print(
                        f"  [reset] ep={episode_count} ran "
                        f"{(step_count - episode_start_step) * CONTROL_DT:.2f}s, "
                        f"reason: {reason}",
                        flush=True,
                    )
                    reset_state(reason)
                    wall_start = time.time() - sim_time

        _finish_action_audit()
        if trace_stream is not None:
            trace_stream.close()
            print(f"[standstill trace] closed {args.standstill_trace_csv}", flush=True)
        print("Viewer closed.")


if __name__ == "__main__":
    main()
