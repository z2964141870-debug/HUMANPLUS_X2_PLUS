# X2 WBT 旧数据资产盘点

版本：Phase0 / 2026-08-09
范围：只读盘点；未重定向、未训练、未删除或修改源数据。这里的“旧 X2”均指官方 AimDK v1.0 canonical contract 建立前的结果，只能作为 A/B 证据。

## 总结

| 数据源 | 当前可用源文件 | 原始下载包 | 旧 X2 产物 | 当前裁决 |
|---|---:|---|---|---|
| AMASS 聚合 Stage-II | 11,798 NPZ；11,796 可读；35.748 h；约 36 GB | 本机未找到原始 tar/zip；解压后的源仍完整存在 | 多代 146/264/317/363/67 集；Pilot58 与 Core24；120 条扩展 | 可重新走官方 X2 GMR；旧产物无 Silver/Gold 认证 |
| PHUMA | 21,731 条 G1 29DOF NPY；约 4.9 GB | 下载归档本机未保留；代码和展开数据仍在 | strict/medium/broad/clean291；native17、扩展24/36/120/123 | 数据合同较好，但仍是 G1→旧 X2；free-root 动态未闭环 |
| BONES-seed / SOMA uniform | 370 条已抽取 BVH；元数据覆盖 142,220 条；约 1.1 GB 抽取集 | `soma_uniform.tar.gz` 已不在原路径 | 344 条 locomotion 全量漏斗、pilot25、多种 root contract | 原始 BVH 有价值；旧 BONES→X2 344/344 未过 strict |

源文件状态的关键结论：**不是“数据都删了”**。AMASS 的 11,798 条展开 Stage-II、PHUMA 的 21,731 条 G1 动作和 BONES 的 370 条已抽取 BVH 都仍在；缺失的是若干最初下载的压缩包，以及 BONES 非 locomotion 的绝大部分 43 GB 归档内容。

## AMASS

- 源根目录：`/home/humanplus/x2_teleop_final/x2_sonic/data/raw`
- 子集计数：ACCAD 252、BMLmovi 1,864、BMLrub 3,061、HDM05 215、KIT 4,232、SFU 44、TotalCapture 37、Transitions 110、sonic_release_cmu 1,983；总计 11,798。
- 源级报告：`/home/humanplus/x2_teleop_final/x2_sonic/x2_gmr_data_rebuild/outputs/amass_source_summary.md`；其 JSON SHA-256 为 `df017e72ace8ca101d3c61f15b609dc4b28fa7e9c8f48d564b2e0f54af825a3d`。
- 旧主要版本：`motion_lib_x2/amass_cmu71_bmlmovi75_x2_146*`、`amass_x2_curriculum_v2_264*`、`v3_317*`、`v4_363*`、`v5_363_kinematic_safe_split`、`v6_rootfoot_clean67_split`；另有 GMR rebuild 的 Core24、Pilot58 和 120 条 locomotion 扩展。
- 已知系统性问题：旧版本建立在非官方/推测性 X2 模型与 body contract 上；旧 KIT 100 Hz 动作曾实际输出约 33 Hz 而非精确 30 Hz；Pilot58 只有结构合同 58/58，Gold=0、Silver=0；120 条扩展 strict=0、near-miss=0，主要卡在 q-step、root/支撑足一致性和滑移。
- 修复结论：exact root-anchor repair 得到 11 条“严格通过”，但部分 root XY 被改到约 3 m，不能训练；regularized repair 仅 3 条通过且不是有效动态行走。证据：`motion_lib_x2/x2_dynamic_clean_gmr_near_miss_repaired_20260716/repair_summary.md` 与 `...repaired_regularized_20260716/repair_summary.md`。
- 哈希：每个 Stage-II NPZ 都可直接获得 SHA-256；旧 Core24 JSON 已包含逐动作 `motion_sha256`。本轮没有为了盘点而读取/哈希完整 36 GB 构造新树哈希。

## PHUMA

- 源根目录：`/home/humanplus/humanoid-GPT/A/sonic_release/PHUMA/data/g1`。
- 来源：DAVIAN-Robotics/PHUMA，本地代码 HEAD `f0306c1687c75aaa6623f5ed16c54f9480bb4f17`；README SHA-256 `3baab22fc6dbaaa077821c8672cf41ef1ffcd9719a96dba12f6ea410a23cc0b3`。
- 数据格式：每条 NPY 是 `root_trans [T,3]`、`root_ori [T,4]`、`dof_pos [T,29]`、`fps`；它是 PHUMA 已经物理筛选/约束到 G1 的机器人动作，不是 X2 原生真值。
- 旧主要版本：`phuma_x2_strict90_50fps`、`medium169`、`broad227`、`hybrid1200_clean291`、`strict89_foundation`；GMR rebuild 还包括 native17、Move_Walk/Emotion_Turn 24、registered36、game-motion120、locomotion123 和 repair/warm-start 分支。
- 已知系统性问题：G1→X2 二次重定向会重新引入形态/接触误差；native17 虽 17/17 格式通过，但尚无 Silver；Move_Walk clip28 的 prescribed-root 跟踪较好，free-root 在 ideal/nominal/delay/noise 四域仍均 root 过低；120 game-motion strict=0，123 locomotion strict=0、near-miss=15。
- 哈希：本地所有 NPY 均可逐文件哈希；本轮诊断面板已记录所选 PHUMA 文件哈希。原下载归档不在，因此无法从本机追溯其 archive SHA。

## BONES-seed / SOMA uniform

- 元数据根：`/home/humanplus/humanoid-GPT/A/sonic_release/bones_seed`。
- 已抽取 BVH：`/home/humanplus/x2_teleop_final/x2_sonic/x2_gmr_data_rebuild/extracted/bones_seed_uniform_locomotion_full_20260717_v2`，共 370 个 BVH；其中旧全量漏斗按元数据选了 344 条 locomotion。
- 元数据版本与哈希：`seed_metadata_v004.parquet`=`7cbe5624...ea97e7d`，`seed_metadata_v004.csv`=`71d09422...9bbc39`，`seed_metadata_v002_temporal_labels.jsonl`=`302da3c3...4d634`（完整值见 JSON 清单）。
- 原 archive：旧报告指向 `/home/humanplus/humanoid-GPT/A/sonic_release/bones_seed/soma_uniform.tar.gz`，当前不存在；因此没有本地 archive SHA，也不能在不重新下载的情况下重新抽取非 locomotion 全集。
- 旧结果：344/344 成功转成 X2 31DOF，但 strict=0、near-miss=0、Silver=0；检测到 1,715 个 DS→SS→DS 周期，仍 344/344 q-step 超门、344/344 支撑段漂移超门、343/344 支撑足滑移超门，slip p50=1.785 m/s。
- 解释边界：这否定的是旧 BONES→X2 contract，不是否定原始 SOMA 动捕；已抽取 BVH 可以重新走官方 X2 pipeline。

## 跨数据源旧结果与不可混淆的事实

- 统一旧审计：`/home/humanplus/x2_teleop_final/x2_sonic/x2_gmr_data_rebuild/outputs/x2_three_source_unified_manifest_20260717.json`，SHA-256 `c880e46384f1b65b57c7374979622d908b5e363b63f933415629fe30c1dee064`。
- 旧三源 certified Gold=0、Silver=0；接触、COM、DCM 均来自旧 X2 FK/MJCF 模型估计，不是真实足底力/COP。
- prescribed-root replay 只说明关节目标可跟踪，不说明浮动基座平衡可执行；因此不能把它升级成 Gold。
- 本轮 Phase0 只把源、版本和失败证据冻结下来。下一步 official-X2 A/B 必须写新目录，不覆盖任何旧 cache。

## 当前数据缺口

1. BONES 原始 `soma_uniform.tar.gz` 缺失，当前只能覆盖 370 条已抽取 BVH；若面板验证新官方 contract 有显著收益，再决定是否重下全包。
2. AMASS/PHUMA/BONES 都没有 X2 真实 GRF、COP 或 centroidal momentum；官方仿真估计必须显式标注为模型量。
3. 旧产物没有与 AimDK v1.0 官方 X2 MJCF/PD 完整绑定的 provenance，不能直接复用为训练 Silver。
4. 数据集级 Merkle/tree hash 尚未建立；逐文件哈希可获得，面板 24 条已经登记 SHA-256。
