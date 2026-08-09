# X2 WBT Phase42：contact-first centroidal teacher

## 假设

不再修 Phase40 joint path；先从水平foot placement与DS→R-SS→DS时序构造地面接触，再求动力学一致COM/GRF，能够隔离原GMR路径的接触几何错误。

## 干预

- A层：25帧COM xyz + official sole sphere GRF，固定DAQP sequential-convex 10轮。
- stance phase的12个sole点使用该phase首帧水平位置并投影z=0；非接触脚没有force变量。
- 硬约束：线动量、线性化零净矩、fz>=0、μ=1摩擦锥、COP凸包、COM边界/速度/加速度。
- 所有量均为official MuJoCo模型估计，不是实机GRF/COP真值。

## 结果

- QP solved：0/10；failure=`daqp returned no solution at outer iteration 1`。对完全相同的144条等式、4926条不等式与边界做HiGHS零目标可行性审计，同样返回`infeasible`；因此不是仅由DAQP数值故障造成。
- dynamics residual：2.27374e-13N；zero-moment residual：171.414N·m。
- friction violation：2.16896N；min normal：15.9619N。
- COM correction XY/Z：0.0000/0.0000m。
- COM speed horizontal/vertical：0.3501/0.2372m/s。
- COM accel horizontal/vertical：4.2063/8.4022m/s²。

## 裁决

**PHASE42_LAYER_A_CENTROIDAL_INFEASIBLE**

固定contact-first centroidal A层未取得硬可行解；按门停止，不用soft reward或调整边界/权重绕过，B/C均不运行。

## 下一步

停止Phase42；报告solver/硬门失败，不进入IK、完整轨迹、physics或PPO。
