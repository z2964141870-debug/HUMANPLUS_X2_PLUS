# X2 WBT Phase40：root-XY structural A/B local certificate

## 假设

Phase39 的25帧局部证书失败，可能主要由冻结 root XY 造成；只释放 root XY 后，同一接触/连续性硬门可能变得可行。

## 干预

- A：直接读取已冻结 Phase39，不重跑。
- B：唯一新增每帧 root XY；相对source欧氏硬界0.20m，水平root acceleration硬门4.0m/s²。
- DAQP、20轮、trust 5mm/0.05rad、active-sphere、contact intent、时间、root orientation、upper/head全部冻结。

## 对照与结果

- A feasible：`False`；stance speed=0.3090m/s；stance distance max=0.001035m。
- B feasible：`False`；stance speed=0.2513m/s；stance distance max=0.000851m。
- B rootXY correction max=0.0063m；root horizontal accel p95/max=[3.999073616172005, 4.000000000000039]m/s²。
- B QP solved=20；failure=`None`；active switches L/R=[1, 0]。
- 执行记录：首次调用在首个QP后因metric读取了不存在的`foot_xy`导出键而中止、未形成结果；仅修正为复用Phase39静态FK后，以完全相同配置完成本次20轮。未据中止结果调参。

## 结论

**PHASE40_ROOT_XY_LOCAL_CERTIFICATE_FAILED**

只释放root XY仍未取得同一25帧局部硬门证书；fixed-rootXY不是单独足以解释Phase39失败的主因。这是否定当前生成器/solver结构，不是X2不可能。

## 下一步

按预注册门停止；不缩放rootXY、不调trust/门、不扩完整轨迹或physics。
