"""DC-PEFT overlay sitecustomize：仅在 DCPEFT_* 环境变量显式激活时生效。"""
try:
    import dcpeft_reward_tap
    dcpeft_reward_tap.install()
except Exception as e:  # 防呆：overlay 失败不得破坏宿主进程
    import sys
    print(f"[dcpeft_overlay] install skipped: {e}", file=sys.stderr)
