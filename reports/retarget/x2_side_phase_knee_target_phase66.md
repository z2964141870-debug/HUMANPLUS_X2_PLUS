# X2 native posture Phase66 — side-by-phase physical knee target

Decision: **FAIL_LOCAL_SIDE_PHASE_DECOMPOSITION_STOP**. Pairing exact: **True**.

| condition | global pitch Δ | active pitch Δ | global support Δ | next support Δ | pass |
| --- | ---: | ---: | ---: | ---: | --- |
| `DS0_left` | +0.000251 | -0.000809 | +0.000917 | +0.001420 | False |
| `DS0_right` | -0.000024 | +0.000289 | -0.000437 | -0.001693 | False |
| `right_swing_left_support_left` | +0.000501 | +0.000201 | +0.000872 | +0.000590 | False |
| `right_swing_left_support_right` | +0.000731 | +0.000881 | -0.000729 | -0.004342 | True |
| `DS_half_left` | +0.000047 | -0.000355 | -0.000331 | +0.000049 | False |
| `DS_half_right` | -0.000748 | -0.000836 | +0.000077 | +0.000021 | False |
| `left_swing_right_support_left` | -0.000891 | -0.000389 | +0.000941 | -0.009678 | False |
| `left_swing_right_support_right` | -0.001218 | -0.001374 | -0.001304 | -0.001866 | False |

Mirrored roles identified: `[]`.

All nonlinear quantiles were recomputed from original same-index samples. No optimizer, checkpoint, ONNX, or deployment backend was produced.
