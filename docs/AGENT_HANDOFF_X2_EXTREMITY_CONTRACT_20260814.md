# Agent 接班任务卡：X2 六链末端目标契约

版本：v1.0  
日期：2026-08-14  
任务状态：纯函数检查与官方 X2 `sole12` 静态探针已通过；尚未接入训练 observation 或真机

## 1. 开工信息

- 项目根目录：`/home/yu/projects/BFM-Zero`
- GitHub 代码写入位置：`/home/yu/projects/BFM-Zero/external/bfm_zero_x2_snapshot`（核心 BFM 源码）；仓库根目录保留脚本、测试、文档与 manifest
- 数据写入位置：`/media/yu/FAFF-E977/data/BFM-Zero`
- 环境：沿用 `/home/yu/miniconda3/envs/x2-sonic-isaaclab`，不得迁移或提交环境目录
- 工作分支：`work/codex-x2-extremity-contract`
- GitHub 基线 commit：`8fe53ba8d58e6732f52edae13ecbf5008a5de611`
- 本地旧 BFM 历史归档分支：`archive/codex-x2-extremity-contract-bfm`（保留原提交 `d55043be103101e11fd450a29073a38470950d3a`）
- 当前任务 commit：接班时运行 `git rev-parse HEAD`，以 GitHub 上该分支最新 commit 为准

接班 Agent 必须先阅读 `/home/yu/projects/AGENT_TASK_CARD.md`，然后运行：

```bash
cd /home/yu/projects/BFM-Zero
git status --short --branch
git rev-parse HEAD
git remote -v
```

若工作树存在非本任务改动，停止写入并先报告，不得覆盖、清理或隐藏历史资产。

## 2. GitHub 必读文件（按顺序）

### A. 项目约束与当前决策

1. `docs/AGENT_HANDOFF_X2_EXTREMITY_CONTRACT_20260814.md`（本卡）
2. `docs/X2_EXTREMITY_CONTRACT_TODO_20260814.md`
3. `docs/X2_OBSERVATION_INTEGRATION_AUDIT_20260815.md`
4. `manifests/x2_observation_dimension_audit_20260815.json`
5. `docs/X2_RESPONSE_CLONE_AUDIT_20260814.md`
6. `manifests/x2_response_clone_audit_20260814.json`
7. `docs/X2_TEACHER_FEASIBILITY_20260814.md`
8. `manifests/x2_teacher_feasibility_20260814.json`

阅读目的：理解为什么本轮停止重复 upper-body disturbance adaptation、为什么不继续并行 MPC，以及当前“15 维下肢腰部策略 + 14 维外部双臂 + 2 维锁头”的部署边界。

### B. 本轮核心代码

9. `external/bfm_zero_x2_snapshot/humanoidverse/x2_extremity_contract.py`
10. `external/bfm_zero_x2_snapshot/humanoidverse/x2_extremity_adapter.py`
11. `tests/test_x2_extremity_contract.py`
12. `tests/test_x2_extremity_adapter.py`
13. `scripts/probe_x2_extremity_contract_v11.py`

阅读目的：理解六链目标的形状、坐标系、四元数约定、31 关节分区以及静态探针的通过条件。不要在没有核对接口的情况下直接改 observation 维度。

### C. 失败链路与可复用工具

12. `external/bfm_zero_x2_snapshot/humanoidverse/x2_response_clone.py`
13. `scripts/audit_x2_response_state_clone_v10.py`
14. `external/bfm_zero_x2_snapshot/humanoidverse/x2_direct_teacher.py`
15. `scripts/search_x2_direct_cem_teacher_v9.py`
16. `external/bfm_zero_x2_snapshot/humanoidverse/x2_teacher_feedback.py`
17. `scripts/screen_x2_state_feedback_teacher_v8.py`
18. `tests/test_x2_response_clone.py`
19. `tests/test_x2_direct_teacher.py`
20. `tests/test_x2_teacher_feedback.py`

阅读目的：复用已经实现的验证框架，同时避免重跑已被证伪的方案。任何新路线都必须说明相对这些失败实验新增了什么可观测信息或可控制自由度。

### D. 接入训练前必须定位的项目文件

接班 Agent 应使用 `rg` 搜索并记录实际路径，不要凭文件名猜测：

```bash
rg -n "X2|sole12|observation|obs_dim|policy_obs|action_scale" external/bfm_zero_x2_snapshot/humanoidverse
rg -n "base_link|torso_link|wrist_roll_link|ankle_roll_link" .
rg -n "left_hip_pitch_joint|left_shoulder_pitch_joint|head_yaw_joint" external/bfm_zero_x2_snapshot
```

运行仓库根目录的脚本或测试时，必须显式加入源码快照路径，避免误导入其他 checkout：

```bash
export PYTHONPATH="$PWD/external/bfm_zero_x2_snapshot${PYTHONPATH:+:$PYTHONPATH}"
```

重点找到并阅读：

- X2 机器人资产配置与 URDF/USD 加载入口；
- 当前任务的 observation 构建函数和 observation dimension 配置；
- 15 维 lower-body action 到 X2 关节的映射；
- 外部双臂目标的注入位置；
- policy checkpoint 的加载和维度校验；
- 训练、评估与 sim2sim 的启动配置。

定位结果应补充到本卡或新 manifest，形成可复查的精确路径清单。

## 3. 接班后的第一阶段任务

1. 只检查仓库、分支、commit 和外部数据目录，不启动机器人相关程序。
2. 运行纯函数检查并记录命令、结果和 commit。
3. 运行只读静态探针，确认六链名称和 31 关节分区；报告写入外部数据目录。
4. 若静态探针失败，只修正契约或资产映射，不进入训练。
5. 若静态探针通过，提出 observation 接入的最小 diff 和旧权重兼容方案，得到确认后再实施。

本次已完成第 2、3 步：

- 纯函数：`9 passed`；
- 静态探针：`PASS_X2_SIX_LINK_CONTRACT_FOR_STATIC_TRACKING_SMOKE`；
- 报告：`/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-15/x2_extremity_contract/x2_extremity_contract_v11.json`；
- 报告 SHA256：`54246ae90a9e51107bd900a251f94737aab49a637511df5f80e111ae87d1dfd7`；
- 训练和部署权限仍为锁定状态。
- 零破坏 adapter 原型已通过 `12 passed`，但尚未接入实际 SONIC backbone。

## 4. 明确的验收门槛

- 六个链节均唯一解析；
- 15+14+2 个关节无重复并完整覆盖目标 31 关节；
- 当前位姿等于目标位姿时特征为零；
- 四元数 `q` 与 `-q` 产生相同旋转误差；
- 单个腕部扰动只影响对应链节特征；
- 日志、报告、checkpoint 和缓存不进入 Git 工作树；
- 未通过上述门槛，不得启动训练，更不得上真机。

## 5. 当前待解决的科研问题

- 衣服输出如何变成带时间语义的未来六链目标；
- 如何显式建模真实 BLE/IMU 链路的延迟、抖动、丢包和漂移；
- observation adapter、分支编码器与从头训练三种方案如何选择；
- 上肢外部控制对下肢平衡策略的扰动如何进入训练分布；
- 六链未来目标相比当前目标和仅腕部目标是否有统计显著收益。

详细优先级见 `docs/X2_EXTREMITY_CONTRACT_TODO_20260814.md`。

## 6. 禁止事项

- 不控制或唤醒真机，不停止/启动机器人系统服务；
- 不把策略输出直接扩成 31 维全身动作；
- 不重跑已明确失败的 Phase55–61 路线，除非提出新的可证伪假设；
- 不提交数据集、日志、视频、checkpoint、环境、密钥或个人设备信息；
- 不删除旧 checkpoint、失败样本、历史报告和可恢复资产；
- 不 force-push，不重写公共历史，不直接向上游 `origin` 推送。

## 7. 结束报告格式

接班 Agent 结束任务时必须报告：

- 项目根目录、代码写入位置、数据写入位置；
- 分支、起止 commit、工作树状态；
- 修改文件与每个文件的作用；
- 执行过的检查/实验命令及结果；
- 外部数据产物、SHA256 和恢复路径；
- 仍未解决的问题、风险与下一步停止条件。
