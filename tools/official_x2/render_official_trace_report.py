#!/usr/bin/env python3
"""Render smooth, reviewable videos from immutable official AimDK traces.

This does not re-simulate or improve a rollout.  It reconstructs the recorded
50 Hz root/joint state in the official X2 MJCF and renders every trace sample
offline.  The resulting video is therefore a visualization of an existing
official-physics trace, with no X11 viewer frame-drop ambiguity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


ISAAC_JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_yaw_joint",
    "right_wrist_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
)

STAGE208_DEFAULT = {name: 0.0 for name in ISAAC_JOINTS}
for _name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
    STAGE208_DEFAULT[_name] = -0.248
for _name in ("left_knee_joint", "right_knee_joint"):
    STAGE208_DEFAULT[_name] = 0.5303
for _name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
    STAGE208_DEFAULT[_name] = -0.2823
for _name in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"):
    STAGE208_DEFAULT[_name] = 0.4
for _name in ("left_elbow_joint", "right_elbow_joint"):
    STAGE208_DEFAULT[_name] = -1.2

FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


class FFmpegWriter:
    """Small dependency-free RGB writer using the system ffmpeg binary."""

    def __init__(self, output: Path, *, width: int, height: int, fps: int):
        self.output = output
        self.width = width
        self.height = height
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-video_size", f"{width}x{height}", "-framerate", str(fps),
                "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
                "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(output),
            ],
            stdin=subprocess.PIPE,
        )

    def append_data(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"invalid RGB frame {frame.shape}/{frame.dtype}")
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def __enter__(self) -> "FFmpegWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        status = self.process.wait()
        if exc_type is None and status != 0:
            raise RuntimeError(f"ffmpeg failed with status {status}: {self.output}")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def root_roll_pitch_from_projected_gravity(gravity: Iterable[float]) -> tuple[float, float]:
    """Recover ZYX roll/pitch from R^T [0,0,-1]."""
    gx, gy, gz = np.asarray(tuple(gravity), dtype=np.float64)
    pitch = math.asin(float(np.clip(gx, -1.0, 1.0)))
    roll = math.atan2(-gy, -gz)
    return roll, pitch


def root_quaternion_wxyz(row: dict) -> np.ndarray:
    roll, pitch = root_roll_pitch_from_projected_gravity(row["obs"][6:9])
    xyzw = Rotation.from_euler(
        "ZYX", [float(row["root_yaw_rad"]), pitch, roll]
    ).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def trace_pitch_rad(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [root_roll_pitch_from_projected_gravity(row["obs"][6:9])[1] for row in rows],
        dtype=np.float64,
    )


def _load_move_trace(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row for row in payload["trace"]
        if row.get("stage") == "move" and len(row.get("obs", ())) == 93
    ]
    if len(rows) < 2:
        raise ValueError(f"no complete move trace in {path}")
    elapsed = np.asarray([float(row["elapsed_s"]) for row in rows])
    if not np.allclose(np.diff(elapsed), 0.02, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"trace is not exact 50 Hz: {path}")
    return payload["summary"], rows


class TraceRenderer:
    def __init__(self, scene: Path, width: int, height: int):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        # The vendor scene defaults to a 640x480 offscreen buffer because its
        # interactive viewer does not need report resolution.  This changes
        # rendering capacity only; it does not alter physics or trace state.
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), width)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), height)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.width = width
        self.height = height
        self.qpos_address = {}
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name and name != "floating_base_joint":
                self.qpos_address[name] = int(self.model.jnt_qposadr[joint_id])
        missing = sorted(set(ISAAC_JOINTS) - set(self.qpos_address))
        if missing:
            raise ValueError(f"official scene lacks trace joints: {missing}")

    def close(self) -> None:
        self.renderer.close()

    def set_row(self, row: dict) -> None:
        qpos = self.data.qpos
        qpos[0:3] = (
            float(row["root_x_m"]),
            float(row["root_y_m"]),
            float(row["root_z_m"]),
        )
        qpos[3:7] = root_quaternion_wxyz(row)
        relative = np.asarray(row["obs"][12:43], dtype=np.float64)
        for index, name in enumerate(ISAAC_JOINTS):
            qpos[self.qpos_address[name]] = STAGE208_DEFAULT[name] + relative[index]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def render(self, camera: mujoco.MjvCamera) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()


def _free_camera(*, lookat: tuple[float, float, float], distance: float,
                 azimuth: float, elevation: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def _draw_banner(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 82), fill=(4, 28, 50, 220))
    draw.text((18, 8), title, font=_font(28), fill=(255, 255, 255, 255))
    draw.text((18, 45), subtitle, font=_font(20), fill=(255, 224, 95, 255))
    draw.rectangle((0, image.height - 38, image.width, image.height), fill=(0, 0, 0, 190))
    draw.text(
        (14, image.height - 32),
        "OFFLINE REPLAY OF RECORDED OFFICIAL AimDK v1.0 PHYSICS TRACE | 50 Hz",
        font=_font(16), fill=(245, 245, 245, 255),
    )
    return np.asarray(image)


def _draw_path_inset(frame: np.ndarray, rows: list[dict], index: int,
                     color: tuple[int, int, int]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    box = (image.width - 232, 92, image.width - 16, 300)
    draw.rounded_rectangle(box, radius=10, fill=(0, 0, 0, 175), outline=(255, 255, 255, 180), width=2)
    points = np.asarray([[row["root_x_m"], row["root_y_m"]] for row in rows], dtype=np.float64)
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 0.25)
    scale = 160.0 / span

    def xy(point: np.ndarray) -> tuple[int, int]:
        # +X is screen-up, +Y is screen-right.
        return (
            int((box[0] + box[2]) * 0.5 + (point[1] - center[1]) * scale),
            int((box[1] + box[3]) * 0.5 - (point[0] - center[0]) * scale),
        )

    full = [xy(point) for point in points]
    if len(full) > 1:
        draw.line(full, fill=(180, 180, 180, 180), width=3)
        draw.line(full[: index + 1], fill=(*color, 255), width=6)
    draw.ellipse((*np.subtract(full[0], (5, 5)), *np.add(full[0], (5, 5))), fill=(70, 255, 120, 255))
    current = full[index]
    draw.ellipse((*np.subtract(current, (6, 6)), *np.add(current, (6, 6))), fill=(*color, 255))
    yaw = float(rows[index]["root_yaw_rad"])
    arrow = (current[0] + int(25 * math.sin(yaw)), current[1] - int(25 * math.cos(yaw)))
    draw.line((current, arrow), fill=(255, 255, 70, 255), width=4)
    draw.text((box[0] + 8, box[1] + 5), "TOP VIEW PATH  +X ↑", font=_font(15), fill=(255, 255, 255, 255))
    return np.asarray(image)


def render_turn_pair(scene: Path, right_trace: Path, left_trace: Path, output: Path) -> dict:
    right_summary, right = _load_move_trace(right_trace)
    left_summary, left = _load_move_trace(left_trace)
    frame_count = min(len(right), len(left))
    right, left = right[:frame_count], left[:frame_count]
    render_right = TraceRenderer(scene, 640, 720)
    render_left = TraceRenderer(scene, 640, 720)
    try:
        all_points = np.asarray(
            [[row["root_x_m"], row["root_y_m"]] for row in right + left], dtype=np.float64
        )
        center = all_points.mean(axis=0)
        camera_right = _free_camera(lookat=(center[0], center[1], 0.0), distance=2.55, azimuth=90, elevation=-82)
        camera_left = _free_camera(lookat=(center[0], center[1], 0.0), distance=2.55, azimuth=90, elevation=-82)
        output.parent.mkdir(parents=True, exist_ok=True)
        with FFmpegWriter(output, width=1280, height=720, fps=50) as writer:
            for index in range(frame_count):
                render_right.set_row(right[index])
                render_left.set_row(left[index])
                a = _draw_banner(
                    render_right.render(camera_right),
                    "RIGHT TURN",
                    f"command wz=+0.15 rad/s | measured yaw {math.degrees(right_summary['move_yaw_progress_rad']):+.1f}°",
                )
                b = _draw_banner(
                    render_left.render(camera_left),
                    "LEFT TURN",
                    f"command wz=-0.09 rad/s | measured yaw {math.degrees(left_summary['move_yaw_progress_rad']):+.1f}°",
                )
                a = _draw_path_inset(a, right, index, (44, 190, 255))
                b = _draw_path_inset(b, left, index, (255, 105, 180))
                writer.append_data(np.concatenate((a, b), axis=1))
    finally:
        render_right.close()
        render_left.close()
    return {
        "output": str(output), "fps": 50, "frame_count": frame_count,
        "right_yaw_deg": math.degrees(right_summary["move_yaw_progress_rad"]),
        "left_yaw_deg": math.degrees(left_summary["move_yaw_progress_rad"]),
    }


def render_straight_side(scene: Path, trace: Path, output: Path) -> dict:
    summary, rows = _load_move_trace(trace)
    renderer = TraceRenderer(scene, 960, 720)
    pitch = trace_pitch_rad(rows)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with FFmpegWriter(output, width=960, height=720, fps=50) as writer:
            for index, row in enumerate(rows):
                renderer.set_row(row)
                camera = _free_camera(
                    lookat=(float(row["root_x_m"]), float(row["root_y_m"]), 0.56),
                    distance=2.0, azimuth=90, elevation=-8,
                )
                frame = _draw_banner(
                    renderer.render(camera),
                    "STRAIGHT WALK — SIDE VIEW",
                    f"measured mean torso pitch {math.degrees(float(pitch.mean())):+.1f}° | negative = backward lean",
                )
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image, "RGBA")
                draw.line((image.width // 2, 110, image.width // 2, 620), fill=(255, 230, 70, 180), width=3)
                draw.text((image.width // 2 + 8, 110), "world vertical", font=_font(17), fill=(255, 230, 70, 255))
                writer.append_data(np.asarray(image))
    finally:
        renderer.close()
    return {
        "output": str(output), "fps": 50, "frame_count": len(rows),
        "mean_pitch_deg": math.degrees(float(pitch.mean())),
        "median_pitch_deg": math.degrees(float(np.median(pitch))),
        "pitch_p05_deg": math.degrees(float(np.quantile(pitch, 0.05))),
        "pitch_p95_deg": math.degrees(float(np.quantile(pitch, 0.95))),
        "move_forward_m": summary["move_forward_displacement_m"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--straight", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "rendering_contract": "offline replay of immutable official AimDK v1.0 50 Hz physical traces; no simulation or state modification",
        "turn_pair": render_turn_pair(
            args.scene, args.right, args.left,
            args.output_dir / "x2_turn_direction_official_trace_smooth.mp4",
        ),
        "straight_side": render_straight_side(
            args.scene, args.straight,
            args.output_dir / "x2_straight_posture_official_trace_smooth.mp4",
        ),
    }
    (args.output_dir / "x2_official_trace_smooth_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
