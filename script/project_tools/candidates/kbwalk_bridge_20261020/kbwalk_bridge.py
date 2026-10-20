#!/usr/bin/env python3
"""kbwalk_bridge.py — laptop-side keyboard/timed intent source for Sonic-X2 v0.2.

Replaces the pad chain (ritual_start_kb.sh already removed the pad bridge
and armed pc2_kplanner_onnx.py with --cmd-bind + --command-watchdog-s 0.3).
This bridge CONNECTS to the planner's SUB on :5563 (planner owns the socket)
and sends `planner_cmd` ZMQ multipart at 20 Hz:
    [b"planner_cmd", b'{"intent":"locomotion","magnitude":"continuous",
      "stick_fwd":0.6,"stick_side":0.0,"stick_yaw":0.0}']
stick_* are normalized intents (not m/s). Zeros request the planner stop
chain (brake pulse + foot alignment), NOT instant power removal.

Safety model (mirrors the author's pad semantics from F01):
  - DEADMAN: hold SPACE. Without deadman every intent is zero.
  - W/S = forward/back (back starts small), A/D = yaw turn.
  - X sends an immediate all-zero command (like releasing the stick).
  - Window focus loss / any internal error / Ctrl-C => zero burst, then exit.
  - Q = request idle (zeros), keep sending zeros 1 s, then exit.
  - The planner's --command-watchdog-s 0.3 stops the robot if this process
    dies; the physical E-STOP stays the authoritative stop.

Two modes:
  --script "walk 2 0.4" "idle 1"   deterministic timed commands (acceptance)
  --interactive                     pygame window, real keydown/keyup + focus

This file never talks ROS and never publishes HAL. Robot evidence goes to
the JSONL log next to it.
"""
import argparse, json, sys, time

CMD_HZ = 20.0
DEAD_KEY = "space"
KEY_MAP = {"w": ("fwd", +1.0), "s": ("fwd", -0.6), "a": ("yaw", +1.0), "d": ("yaw", -1.0)}


class IntentState:
    """Pure key/intent state machine; unit-tested, no GUI deps."""
    def __init__(self):
        self.keys = set()
        self.focused = True

    def key(self, name, down):
        name = name.lower()
        if down:
            self.keys.add(name)
        else:
            self.keys.discard(name)

    def focus(self, focused):
        self.focused = bool(focused)

    def reset(self):
        self.keys.clear()

    def intent(self):
        """Return the (fwd, side, yaw) normalized intent under current state."""
        if not self.focused or DEAD_KEY not in self.keys:
            return (0.0, 0.0, 0.0)
        fwd = yaw = 0.0
        for k, (axis, val) in KEY_MAP.items():
            if k in self.keys:
                if axis == "fwd":
                    fwd += val
                else:
                    yaw += val
        fwd = max(-1.0, min(1.0, fwd))
        yaw = max(-1.0, min(1.0, yaw))
        return (fwd, 0.0, yaw)


def encode_cmd(fwd, side, yaw, intent="locomotion"):
    payload = {
        "intent": intent, "magnitude": "continuous",
        "stick_fwd": round(float(fwd), 3),
        "stick_side": round(float(side), 3),
        "stick_yaw": round(float(yaw), 3),
    }
    return [b"planner_cmd", json.dumps(payload).encode("utf-8")]


def parse_script(items, default_scale=1.0):
    """Parse ['walk 2 0.4', 'turn 1 0.3', 'idle 1'] into (duration, fwd, yaw) steps."""
    steps = []
    for it in items:
        parts = it.split()
        if not parts:
            continue
        op = parts[0].lower()
        if op == "walk":
            dur = float(parts[1]) if len(parts) > 1 else 1.0
            fwd = float(parts[2]) if len(parts) > 2 else 0.4
            steps.append((dur, fwd, 0.0))
        elif op == "turn":
            dur = float(parts[1]) if len(parts) > 1 else 1.0
            yaw = float(parts[2]) if len(parts) > 2 else 0.3
            steps.append((dur, 0.0, yaw))
        elif op == "idle":
            dur = float(parts[1]) if len(parts) > 1 else 1.0
            steps.append((dur, 0.0, 0.0))
        else:
            raise ValueError(f"unknown script op: {op!r}")
    if not steps:
        raise ValueError("empty script")
    return steps


class Sender:
    """ZMQ PUB connect + JSONL evidence log. Falls back to dry mode w/o zmq."""
    def __init__(self, endpoint, log_path, dry=False):
        self.dry = dry
        self.log = open(log_path, "a", buffering=1)
        if dry:
            self.sock = None
            return
        import zmq
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.SNDHWM, 8)
        self.sock.connect(endpoint)
        time.sleep(0.5)  # PUB connect handshake; first frames may drop

    def send(self, fwd, side, yaw, note=""):
        parts = encode_cmd(fwd, side, yaw)
        if self.sock is not None:
            self.sock.send_multipart(parts)
        self.log.write(json.dumps({
            "utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            "mono_s": round(time.monotonic(), 6),
            "cmd": json.loads(parts[1].decode()), "note": note}) + "\n")

    def close(self):
        try:
            if self.sock is not None:
                self.sock.close(0)
        finally:
            self.log.close()


def run_script(sender, steps):
    """Timed-command mode: deadman-equivalent intent for exact durations."""
    period = 1.0 / CMD_HZ
    for dur, fwd, yaw in steps:
        end = time.monotonic() + dur
        while time.monotonic() < end:
            sender.send(fwd, 0.0, yaw, note="script")
            time.sleep(period)
    # stop chain: hold zeros longer than the brake pulse (0.55 s default)
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        sender.send(0.0, 0.0, 0.0, note="script-stop")
        time.sleep(period)


def run_interactive(sender):
    """pygame window mode: real keydown/keyup + focus loss => zeros."""
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((420, 200))
    pygame.display.set_caption("kbwalk — SPACE=deadman W/S/A/D — X=zero Q=quit")
    st = IntentState()
    clock = pygame.time.Clock()
    period = 1.0 / CMD_HZ
    quitting = False
    while not quitting:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                quitting = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    quitting = True
                elif ev.key == pygame.K_x:
                    st.reset()
                else:
                    st.key(pygame.key.name(ev.key), True)
            elif ev.type == pygame.KEYUP:
                st.key(pygame.key.name(ev.key), False)
            elif ev.type == pygame.ACTIVEEVENT:
                st.focus(bool(ev.gain))
        if not st.focused:
            st.reset()
        fwd, side, yaw = st.intent()
        sender.send(fwd, side, yaw, note="live")
        screen.fill((24, 24, 24))
        pygame.display.set_caption(
            f"kbwalk fwd={fwd:+.1f} yaw={yaw:+.1f} deadman={'ON' if st.focused and DEAD_KEY in st.keys else 'OFF'}")
        pygame.display.flip()
        clock.tick(CMD_HZ)
    # graceful: zeros for the stop chain, then exit (planner watchdog covers us)
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        sender.send(0.0, 0.0, 0.0, note="exit-stop")
        time.sleep(period)
    pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5563",
                    help="planner_cmd SUB endpoint (planner binds 5563; tunnel -L to SoC1)")
    ap.add_argument("--log", default="kbwalk_bridge.jsonl")
    ap.add_argument("--script", nargs="*", default=None,
                    help="timed steps e.g. --script 'idle 1' 'walk 2 0.4' 'idle 2'")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--dry", action="store_true", help="no ZMQ, evidence log only")
    args = ap.parse_args()
    sender = Sender(args.endpoint, args.log, dry=args.dry)
    ok = True
    try:
        if args.script is not None:
            run_script(sender, parse_script(args.script))
        elif args.interactive:
            run_interactive(sender)
        else:
            ap.error("choose --script ... or --interactive")
    except KeyboardInterrupt:
        ok = False
    finally:
        try:
            end = time.monotonic() + 1.0
            while time.monotonic() < end:
                sender.send(0.0, 0.0, 0.0, note="finally-zero")
                time.sleep(1.0 / CMD_HZ)
        finally:
            sender.close()
    return 0 if ok else 130


if __name__ == "__main__":
    sys.exit(main())
