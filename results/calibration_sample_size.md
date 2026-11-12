# How many healthy episodes a calibration needs

**SUPPORTING / diagnostic evidence.** Replaces the uncommitted curve behind `_MIN_HEALTHY = 30` and `_RECOMMENDED_HEALTHY = 80`. No published number depends on this; it exists to choose the sample sizes the final evaluation will be built on, before that evaluation is collected.

A single untouched test set of 100 was drawn first and is never fitted or thresholded on; every row below is scored against it. Channels ('e', 'm'), budget 5%, 10 replicates per cell with independent fit and calibration draws.

Dispersion is `theta p90 / theta p10` across replicates: how far the same procedure moves on nothing but which healthy episodes it happened to see.

| fit N | calibration N | replicates | theta p10 | theta median | theta p90 | p90/p10 | mean test FA | worst test FA | replicates within budget |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 10 | 10 | 10.2 | 63.7 | 351.1 | 34.3x | 0.105 | 0.300 | 20% |
| 20 | 20 | 10 | 29.9 | 215.1 | 9.449e+06 | 316113.8x | 0.020 | 0.070 | 80% |
| 20 | 40 | 10 | 22.8 | 129.2 | 6.3e+06 | 276654.4x | 0.026 | 0.090 | 80% |
| 20 | 80 | 10 | 27.9 | 145.9 | 6.299e+06 | 225895.3x | 0.026 | 0.090 | 80% |
| 40 | 10 | 10 | 14.2 | 61.2 | 123 | 8.7x | 0.065 | 0.180 | 40% |
| 40 | 20 | 10 | 16.4 | 61.2 | 323.8 | 19.7x | 0.045 | 0.120 | 60% |
| 40 | 40 | 10 | 26.3 | 59.1 | 276.8 | 10.5x | 0.042 | 0.100 | 60% |
| 40 | 80 | 10 | 35.4 | 55.6 | 161.5 | 4.6x | 0.041 | 0.080 | 50% |
| 80 | 10 | 10 | 6.5 | 34.5 | 89.7 | 13.8x | 0.108 | 0.280 | 30% |
| 80 | 20 | 10 | 16.6 | 67.2 | 120.3 | 7.3x | 0.050 | 0.140 | 50% |
| 80 | 40 | 10 | 27.8 | 62.7 | 68.83 | 2.5x | 0.044 | 0.060 | 60% |
| 80 | 80 | 10 | 31.1 | 44.5 | 69.82 | 2.2x | 0.049 | 0.080 | 50% |
| 140 | 10 | 10 | 23.2 | 33.2 | 100.5 | 4.3x | 0.056 | 0.090 | 40% |
| 140 | 20 | 10 | 23.2 | 57.9 | 140.2 | 6.0x | 0.049 | 0.090 | 50% |
| 140 | 40 | 10 | 27.5 | 70.3 | 145.4 | 5.3x | 0.031 | 0.090 | 70% |
| 140 | 80 | 10 | 27.4 | 70.2 | 148.8 | 5.4x | 0.029 | 0.050 | 100% |

A large `theta p90` beside a low false-alarm rate is not a good row. A threshold that has run away alarms on nothing and earns a false-alarm rate of zero by refusing to detect anything either, so the two columns are only meaningful read together: the most extreme draw here is fit 20 / calibration 20, which picked theta = 9.45e+07 and then false alarmed on 0% of the untouched healthy set.

## What a false-alarm claim needs from the test set

Independent of the monitor: with `k` false alarms in `n` untouched healthy episodes, this is the exact 95% interval. It is what decides the final healthy test size before collection.

| untouched healthy N | observed FA | 95% CI | supports a <=5% claim |
|---|---|---|---|
| 12 | 0.000 (0) | 0.000-0.265 | no |
| 12 | 0.000 (0) | 0.000-0.265 | no |
| 12 | 0.083 (1) | 0.002-0.385 | no |
| 24 | 0.000 (0) | 0.000-0.142 | no |
| 24 | 0.000 (0) | 0.000-0.142 | no |
| 24 | 0.042 (1) | 0.001-0.211 | no |
| 40 | 0.000 (0) | 0.000-0.088 | no |
| 40 | 0.025 (1) | 0.001-0.132 | no |
| 40 | 0.050 (2) | 0.006-0.169 | no |
| 59 | 0.000 (0) | 0.000-0.061 | no |
| 59 | 0.017 (1) | 0.000-0.091 | no |
| 59 | 0.051 (3) | 0.011-0.141 | no |
| 80 | 0.000 (0) | 0.000-0.045 | yes |
| 80 | 0.025 (2) | 0.003-0.087 | no |
| 80 | 0.050 (4) | 0.014-0.123 | no |
| 100 | 0.000 (0) | 0.000-0.036 | yes |
| 100 | 0.020 (2) | 0.002-0.070 | no |
| 100 | 0.050 (5) | 0.016-0.113 | no |
| 120 | 0.000 (0) | 0.000-0.030 | yes |
| 120 | 0.017 (2) | 0.002-0.059 | no |
| 120 | 0.050 (6) | 0.019-0.106 | no |
| 160 | 0.000 (0) | 0.000-0.023 | yes |
| 160 | 0.019 (3) | 0.004-0.054 | no |
| 160 | 0.050 (8) | 0.022-0.096 | no |
| 200 | 0.000 (0) | 0.000-0.018 | yes |
| 200 | 0.020 (4) | 0.005-0.050 | no |
| 200 | 0.050 (10) | 0.024-0.090 | no |
| 300 | 0.000 (0) | 0.000-0.012 | yes |
| 300 | 0.020 (6) | 0.007-0.043 | yes |
| 300 | 0.050 (15) | 0.028-0.081 | no |
