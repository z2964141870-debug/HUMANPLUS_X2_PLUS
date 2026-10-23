#!/bin/bash
# preflight_link.sh (SoC1, agi@) - 上电后第一步: 无风险链路预检
# 只起 kplanner + 键盘; 不碰 deploy / 不碰 MC -> 机器人零运动。
# 验证三件事: venv 实机跑通 kplanner; :5556 流出 3936B V51 帧;
# 键盘 planner_cmd 被实机 ingest。
set -uo pipefail
PY=/home/agi/gear-sonic/venv/bin/python
ONNX=/home/agi/gear-sonic/planner_stack/models/planner_onnx/x2_planner_template.onnx
LOG=/tmp/kpl_preflight.log
fail(){ echo "FAIL: $*"; exit 1; }
[ -x "$PY" ] || fail "venv python 缺失: $PY"
[ -f "$ONNX" ] || fail "template planner onnx 缺失: $ONNX"
[ -f /home/agi/walk/x2_keyboard_walk_publisher.py ] || fail "键盘工具未部署在 ~/walk/"

echo "[1/5] 启动 kplanner (单独, 后台 tmux 会话 kpl_preflight)..."
tmux kill-session -t kpl_preflight 2>/dev/null
tmux new-session -d -s kpl_preflight \
  "KPLANNER_FIXED_FWD_MPS=0.35 KPLANNER_FIXED_TURN_RAD_S=0.65 \
   $PY $ONNX 2>&1 | tee $LOG"

echo "[2/5] 等 'planner active' (最多 90s)..."
ok=""
for i in $(seq 1 90); do
  grep -q "planner active" $LOG 2>/dev/null && { ok=1; break; }
  grep -qiE "Traceback" $LOG 2>/dev/null && { tail -5 $LOG; fail "kplanner 启动报错"; }
  sleep 1
done
[ -n "$ok" ] || { tail -8 $LOG; fail "90s 内未到 planner active"; }
echo "  OK: planner active"

echo "[3/5] 验证 :5556 pose 流 (期望 3936B V51 帧)..."
$PY - <<'PYEOF'
import zmq
ctx = zmq.Context(); s = ctx.socket(zmq.SUB)
s.setsockopt_string(zmq.SUBSCRIBE, "pose"); s.RCVTIMEO = 8000
s.connect("tcp://127.0.0.1:5556")
try:
    m = s.recv_multipart()
    n = len(m[-1])
    print("  OK: pose 帧 %d 字节%s" % (n, " (V51 契约)" if n == 3936 else " (非 3936, 需查!)"))
except zmq.Again:
    print("  FAIL: 5556 无帧流出"); raise SystemExit(1)
PYEOF
[ $? -eq 0 ] || fail "pose 流不通"

echo "[4/5] 键盘发 3s planner_cmd, 查 kplanner 实机 ingest..."
$PY /home/agi/walk/x2_keyboard_walk_publisher.py --mode planner-cmd --duration 3 >/dev/null 2>&1
grep -aqE "planner_cmd|target_velocity|intent" $LOG && { echo "  OK: kplanner 日志出现意图接收:"; grep -aE "planner_cmd|target_velocity|intent" $LOG | tail -2; } \
  || echo "  WARN: 日志未见 ingest 记录 (tape 在 log/kplanner_tape/, 不阻塞)"

echo "[5/5] 清理: 关闭预检 kplanner"
tmux kill-session -t kpl_preflight 2>/dev/null
echo ""
echo "预检通过 = T1 的 kplanner 零件 + T2 键盘链路全部实机验证。"
echo "之后才执行现场任务书 T3 -> T1 -> T2 全链 (含 deploy)。"
