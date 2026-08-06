"""DC-PEFT Phase 1: RewardManager 逐 term 记录 tap（overlay，不改旧工程文件）。

激活方式：把本文件所在目录加入 PYTHONPATH（内含 sitecustomize.py），并设置
环境变量 DCPEFT_REWARD_TAP=<输出 .pt 路径>。仅当 isaaclab.managers.reward_manager
被导入时才打补丁，其他 python 进程不受影响。

记录内容（每次 RewardManager.compute 调用一步）：
- step_reward: (num_envs, num_terms) = raw_term_value * weight（即 _step_reward，value/dt）
- scalar: (num_envs,) compute() 的原始返回值（value 累加，含 dt）
- dt、term_names、term_weights 一次性记录
输出：torch.save dict 到 DCPEFT_REWARD_TAP。
"""
import importlib.abc
import importlib.machinery
import os
import sys

_TARGET = "isaaclab.managers.reward_manager"


def _install_patch(module):
    import atexit
    import torch

    out_path = os.environ["DCPEFT_REWARD_TAP"]
    RewardManager = module.RewardManager
    orig_compute = RewardManager.compute
    log = {"step_reward": [], "scalar": [], "meta": None}

    def tapped_compute(self, dt):
        ret = orig_compute(self, dt)
        log["step_reward"].append(self._step_reward.detach().clone().cpu())
        log["scalar"].append(ret.detach().clone().cpu())
        if log["meta"] is None:
            log["meta"] = {
                "dt": float(dt),
                "term_names": list(self._term_names),
                "term_weights": [float(c.weight) for c in self._term_cfgs],
            }
        return ret

    RewardManager.compute = tapped_compute

    def flush():
        if not log["step_reward"]:
            return
        torch.save(
            {
                "step_reward": torch.stack(log["step_reward"]),  # (T, num_envs, num_terms)
                "scalar": torch.stack(log["scalar"]),            # (T, num_envs)
                **log["meta"],
            },
            out_path,
        )
        print(f"[dcpeft_reward_tap] saved {len(log['step_reward'])} steps -> {out_path}",
              file=sys.stderr)

    atexit.register(flush)
    print(f"[dcpeft_reward_tap] RewardManager.compute patched, output={out_path}",
          file=sys.stderr)


class _TapFinder(importlib.abc.MetaPathFinder):
    """在目标模块首次正常导入完成后打补丁。"""

    def __init__(self):
        self._busy = False

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_and_patch(module):
            orig_exec(module)
            _install_patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def install():
    if os.environ.get("DCPEFT_REWARD_TAP"):
        if _TARGET in sys.modules:
            _install_patch(sys.modules[_TARGET])
        else:
            sys.meta_path.insert(0, _TapFinder())
