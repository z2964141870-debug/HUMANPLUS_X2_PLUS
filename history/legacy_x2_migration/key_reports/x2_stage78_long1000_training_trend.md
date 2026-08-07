# X2 PPO Training Trend

- rows: 1000
- iterations: 1–1000
- window: 250

This report only summarizes training rollouts. It does not promote a checkpoint; promotion requires deterministic contact-aware evaluation.

| iterations | reward ↑ | reward/step ↑ | length ↑ | root pos/orientation ↓ | root velocity ↓ | body pos/velocity ↓ | wrist L/R m ↓ | term ori/EE/foot ↓ | KL |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | ---: |
| 1–250 | 10.4985 | 0.07318 | 142.56 | 0.3044/0.1769 | 0.4158 | 0.0548/0.5197 | 0.0468/0.0555 | 0.002/0.000/0.078 | 0.00130 |
| 251–500 | 11.2697 | 0.07360 | 152.79 | 0.3094/0.1692 | 0.3880 | 0.0542/0.4960 | 0.0473/0.0549 | 0.001/0.000/0.049 | 0.00130 |
| 501–750 | 11.6296 | 0.07426 | 156.39 | 0.3374/0.1513 | 0.4010 | 0.0527/0.5061 | 0.0429/0.0529 | 0.000/0.000/0.036 | 0.00120 |
| 751–1000 | 11.4374 | 0.07399 | 154.26 | 0.3579/0.1663 | 0.4199 | 0.0540/0.5198 | 0.0429/0.0554 | 0.002/0.000/0.048 | 0.00148 |
