# A goal-consistency channel for drift the behavioural channels miss

**SUPPORTING / diagnostic evidence, development data only.** These corpora were spent when their interior was opened. Nothing below is a performance claim, and the pre-fix baseline (`results/framework_generalized_monitor_report.md`) stands unchanged.

`normalized_fusion[e+m]/rate/v2` is held fixed: statistic, normalization, three-fold holdout and the 5% budget are not reopened. Only the fused channel set varies, so a difference is attributable to the channel rather than to a recalibration. The goal channel is scaled and thresholded through the same healthy-only path as the others and has no threshold of its own.

Every figure is the mean of 40 paired development splits, reported with its two blocks of 20 so a result that moved between blocks cannot be read as settled.

## Arms

| dataset | arm | admission | detection | scoreable FA | goal_drift (injected) | goal_drift (realized) | alarms on ignored hijacks |
|---|---|---|---|---|---|---|---|
| langgraph7b_real2 | e+m | task verifier must pass | 0.536 | 0.057 | 0.35/6 = 0.058 | 0.35/3 = 0.117 | 0.00/3 |
| langgraph7b_real2 | e+m+c | task verifier must pass | 0.404 | 0.046 | 1.65/6 = 0.275 | 1.65/3 = 0.550 | 0.00/3 |
| langgraph7b_real2 | c | task verifier must pass | 0.215 | 0.049 | 2.10/6 = 0.350 | 2.05/3 = 0.683 | 0.05/3 |
| autogen7b_real2 | e+m | task verifier must pass | 0.573 | 0.065 | 1.60/6 = 0.267 | 1.38/3 = 0.458 | 0.23/3 |
| autogen7b_real2 | e+m+c | task verifier must pass | 0.430 | 0.061 | 2.25/6 = 0.375 | 2.12/3 = 0.708 | 0.12/3 |
| autogen7b_real2 | c | task verifier must pass | 0.293 | 0.071 | 2.88/6 = 0.479 | 2.50/3 = 0.833 | 0.38/3 |
| real_research7b_long_drift | e+m | collection rule only  no registered verifier | 0.079 | 0.073 | 9.47/120 = 0.079 | 5.95/74 = 0.080 | 3.52/46 |
| real_research7b_long_drift | e+m+c | collection rule only  no registered verifier | 0.148 | 0.071 | 17.77/120 = 0.148 | 15.20/74 = 0.205 | 2.58/46 |
| real_research7b_long_drift | c | collection rule only  no registered verifier | 0.164 | 0.066 | 19.70/120 = 0.164 | 18.90/74 = 0.255 | 0.80/46 |

## Paired against the frozen protocol

Holm-corrected within each deployment, because several metrics are tested per deployment.

| dataset | arm | metric | mean | delta | blocks | Wilcoxon p | Holm p |
|---|---|---|---|---|---|---|---|
| autogen7b_real2 | c | detection_rate | 0.293 | -0.279 | 0.267;0.320 | 0.0000 | 0.0000 |
| autogen7b_real2 | c | scoreable_fa_rate | 0.071 | +0.006 | 0.048;0.093 | 0.8604 | 1.0000 |
| autogen7b_real2 | c | drift_det_realized | 0.833 | +0.375 | 0.867;0.800 | 0.0001 | 0.0001 |
| autogen7b_real2 | e+m+c | detection_rate | 0.430 | -0.142 | 0.457;0.404 | 0.0069 | 0.0069 |
| autogen7b_real2 | e+m+c | scoreable_fa_rate | 0.061 | -0.003 | 0.053;0.069 | 0.5661 | 1.0000 |
| autogen7b_real2 | e+m+c | drift_det_realized | 0.708 | +0.250 | 0.700;0.717 | 0.0009 | 0.0009 |
| langgraph7b_real2 | c | detection_rate | 0.215 | -0.322 | 0.185;0.244 | 0.0000 | 0.0000 |
| langgraph7b_real2 | c | scoreable_fa_rate | 0.049 | -0.008 | 0.040;0.059 | 0.5752 | 1.0000 |
| langgraph7b_real2 | c | drift_det_realized | 0.683 | +0.567 | 0.650;0.717 | 0.0000 | 0.0000 |
| langgraph7b_real2 | e+m+c | detection_rate | 0.404 | -0.132 | 0.379;0.429 | 0.0107 | 0.0107 |
| langgraph7b_real2 | e+m+c | scoreable_fa_rate | 0.046 | -0.011 | 0.046;0.046 | 0.6922 | 1.0000 |
| langgraph7b_real2 | e+m+c | drift_det_realized | 0.550 | +0.433 | 0.617;0.483 | 0.0000 | 0.0000 |
| real_research7b_long_drift | c | detection_rate | 0.164 | +0.085 | 0.167;0.161 | 0.0003 | 0.0003 |
| real_research7b_long_drift | c | scoreable_fa_rate | 0.066 | -0.007 | 0.060;0.072 | 0.3255 | 0.6509 |
| real_research7b_long_drift | c | drift_det_realized | 0.255 | +0.175 | 0.257;0.253 | 0.0000 | 0.0000 |
| real_research7b_long_drift | e+m+c | detection_rate | 0.148 | +0.069 | 0.134;0.162 | 0.0000 | 0.0001 |
| real_research7b_long_drift | e+m+c | scoreable_fa_rate | 0.071 | -0.002 | 0.071;0.071 | 0.9137 | 0.9137 |
| real_research7b_long_drift | e+m+c | drift_det_realized | 0.205 | +0.125 | 0.175;0.236 | 0.0000 | 0.0000 |

## Per failure class

A channel that bought goal_drift by raising the threshold would show it here as a loss on the other classes.

| dataset | arm | class | n | detection |
|---|---|---|---|---|
| langgraph7b_real2 | e+m | context_corruption | 6 | 0.721 |
| langgraph7b_real2 | e+m | goal_drift | 6 | 0.058 |
| langgraph7b_real2 | e+m | looping | 6 | 0.917 |
| langgraph7b_real2 | e+m | tool_cascade | 6 | 0.450 |
| langgraph7b_real2 | e+m+c | context_corruption | 6 | 0.504 |
| langgraph7b_real2 | e+m+c | goal_drift | 6 | 0.275 |
| langgraph7b_real2 | e+m+c | looping | 6 | 0.621 |
| langgraph7b_real2 | e+m+c | tool_cascade | 6 | 0.217 |
| langgraph7b_real2 | c | context_corruption | 6 | 0.246 |
| langgraph7b_real2 | c | goal_drift | 6 | 0.350 |
| langgraph7b_real2 | c | looping | 6 | 0.212 |
| langgraph7b_real2 | c | tool_cascade | 6 | 0.050 |
| autogen7b_real2 | e+m | context_corruption | 6 | 0.654 |
| autogen7b_real2 | e+m | goal_drift | 6 | 0.267 |
| autogen7b_real2 | e+m | looping | 6 | 0.954 |
| autogen7b_real2 | e+m | tool_cascade | 5 | 0.385 |
| autogen7b_real2 | e+m+c | context_corruption | 6 | 0.371 |
| autogen7b_real2 | e+m+c | goal_drift | 6 | 0.375 |
| autogen7b_real2 | e+m+c | looping | 6 | 0.733 |
| autogen7b_real2 | e+m+c | tool_cascade | 5 | 0.205 |
| autogen7b_real2 | c | context_corruption | 6 | 0.087 |
| autogen7b_real2 | c | goal_drift | 6 | 0.479 |
| autogen7b_real2 | c | looping | 6 | 0.471 |
| autogen7b_real2 | c | tool_cascade | 5 | 0.105 |
| real_research7b_long_drift | e+m | goal_drift | 120 | 0.079 |
| real_research7b_long_drift | e+m+c | goal_drift | 120 | 0.148 |
| real_research7b_long_drift | c | goal_drift | 120 | 0.164 |

## Which channel wins the fusion

The goal channel is only useful if the max actually selects it when it is the informed one.

| dataset | arm | e | m | c |
|---|---|---|---|---|
| langgraph7b_real2 | e+m | 0.677 | 0.323 | - |
| langgraph7b_real2 | e+m+c | 0.547 | 0.262 | 0.191 |
| langgraph7b_real2 | c | - | - | 1.000 |
| autogen7b_real2 | e+m | 0.779 | 0.221 | - |
| autogen7b_real2 | e+m+c | 0.536 | 0.174 | 0.290 |
| autogen7b_real2 | c | - | - | 1.000 |
| real_research7b_long_drift | e+m | 0.702 | 0.298 | - |
| real_research7b_long_drift | e+m+c | 0.501 | 0.186 | 0.314 |
| real_research7b_long_drift | c | - | - | 1.000 |

## Length strata

| dataset | arm | band | n healthy | FA | n injected | detection | median alarm step |
|---|---|---|---|---|---|---|---|
| langgraph7b_real2 | e+m | 4-5 | 9 | 0.000 | 4 | 0.250 | 3.0 |
| langgraph7b_real2 | e+m | 6-7 | 21 | 0.000 | 10 | 0.100 | 3.0 |
| langgraph7b_real2 | e+m | 8-99 | 5 | 0.000 | 10 | 0.700 | 5.0 |
| langgraph7b_real2 | e+m+c | 4-5 | 9 | 0.111 | 4 | 0.250 | 3.0 |
| langgraph7b_real2 | e+m+c | 6-7 | 21 | 0.095 | 10 | 0.300 | 5.0 |
| langgraph7b_real2 | e+m+c | 8-99 | 5 | 0.200 | 10 | 0.800 | 5.0 |
| langgraph7b_real2 | c | 4-5 | 9 | 0.111 | 4 | 0.000 | nan |
| langgraph7b_real2 | c | 6-7 | 21 | 0.095 | 10 | 0.400 | 5.5 |
| langgraph7b_real2 | c | 8-99 | 5 | 0.200 | 10 | 0.400 | 5.0 |
| autogen7b_real2 | e+m | 4-5 | 8 | 0.000 | 10 | 0.300 | 3.0 |
| autogen7b_real2 | e+m | 6-7 | 10 | 0.000 | 5 | 0.400 | 3.0 |
| autogen7b_real2 | e+m | 8-99 | 11 | 0.182 | 8 | 0.750 | 4.0 |
| autogen7b_real2 | e+m+c | 4-5 | 8 | 0.000 | 10 | 0.300 | 3.0 |
| autogen7b_real2 | e+m+c | 6-7 | 10 | 0.000 | 5 | 0.600 | 3.0 |
| autogen7b_real2 | e+m+c | 8-99 | 11 | 0.182 | 8 | 0.875 | 5.0 |
| autogen7b_real2 | c | 4-5 | 8 | 0.000 | 10 | 0.100 | 3.0 |
| autogen7b_real2 | c | 6-7 | 10 | 0.000 | 5 | 0.400 | 3.5 |
| autogen7b_real2 | c | 8-99 | 11 | 0.182 | 8 | 0.625 | 3.0 |
| real_research7b_long_drift | e+m | 4-5 | 0 | nan | 1 | 0.000 | nan |
| real_research7b_long_drift | e+m | 6-7 | 0 | nan | 0 | nan | nan |
| real_research7b_long_drift | e+m | 8-99 | 36 | 0.028 | 119 | 0.000 | nan |
| real_research7b_long_drift | e+m+c | 4-5 | 0 | nan | 1 | 1.000 | 3.0 |
| real_research7b_long_drift | e+m+c | 6-7 | 0 | nan | 0 | nan | nan |
| real_research7b_long_drift | e+m+c | 8-99 | 36 | 0.028 | 119 | 0.210 | 8.0 |
| real_research7b_long_drift | c | 4-5 | 0 | nan | 1 | 1.000 | 3.0 |
| real_research7b_long_drift | c | 6-7 | 0 | nan | 0 | nan | nan |
| real_research7b_long_drift | c | 8-99 | 36 | 0.028 | 119 | 0.210 | 7.0 |

## Is the loss on the other classes masking, or budget?

Adding a channel to a max fusion held to a fixed false-alarm budget raises the threshold, and a higher threshold costs detections that had nothing to do with the new channel. The third row separates the two explanations by scoring the three-channel monitor at the two-channel monitor's threshold.

| dataset | condition | context_corruption | goal_drift | looping | tool_cascade | scoreable FA |
|---|---|---|---|---|---|---|
| langgraph7b_real2 | e+m at its own theta | 0.711 | 0.067 | 0.900 | 0.356 | 0.053 |
| langgraph7b_real2 | e+m+c at its own theta | 0.533 | 0.311 | 0.578 | 0.189 | 0.055 |
| langgraph7b_real2 | e+m+c at the e+m theta | 0.711 | 0.433 | 0.922 | 0.378 | 0.099 |
| autogen7b_real2 | e+m at its own theta | 0.689 | 0.256 | 1.000 | 0.360 | 0.037 |
| autogen7b_real2 | e+m+c at its own theta | 0.511 | 0.322 | 0.922 | 0.227 | 0.039 |
| autogen7b_real2 | e+m+c at the e+m theta | 0.689 | 0.356 | 1.000 | 0.360 | 0.060 |
| real_research7b_long_drift | e+m at its own theta | nan | 0.055 | nan | nan | 0.093 |
| real_research7b_long_drift | e+m+c at its own theta | nan | 0.114 | nan | nan | 0.072 |
| real_research7b_long_drift | e+m+c at the e+m theta | nan | 0.262 | nan | nan | 0.170 |

Median theta(e+m+c) / theta(e+m): autogen7b_real2 1.00, langgraph7b_real2 1.15, real_research7b_long_drift 4.68.

At the fold sizes these corpora allow, `pick_threshold` returns the single most extreme healthy episode: with 17 threshold episodes the order-statistic floor 1/(n+1) = 0.056 already exceeds the 5% budget, so no healthy episode sits above theta and one episode sets it. That is why a third channel moves theta at all. The protocol's own floor of 80 threshold episodes puts theta at the 4th largest of 80, and the untouched test size of 300 at the 15th of 300; neither exists yet.


## Runtime cost

Per-step scoring, measured in the same pass. An accuracy gain bought by making the monitor materially heavier is not free. Wall-clock is a property of the machine and of what else it was doing: read the RATIO between arms measured in one pass, not the absolute microseconds.

| dataset | arm | p50 us/step | p95 us/step |
|---|---|---|---|
| langgraph7b_real2 | e+m | 167.2 | 176.4 |
| langgraph7b_real2 | e+m+c | 255.4 | 335.8 |
| langgraph7b_real2 | c | 77.1 | 82.3 |
| autogen7b_real2 | e+m | 165.9 | 212.1 |
| autogen7b_real2 | e+m+c | 184.3 | 398.5 |
| autogen7b_real2 | c | 49.3 | 55.1 |
| real_research7b_long_drift | e+m | 169.8 | 301.1 |
| real_research7b_long_drift | e+m+c | 253.2 | 266.4 |
| real_research7b_long_drift | c | 78.0 | 91.5 |

Scoring a third channel is a third ESN, so the per-step cost rises roughly in proportion -- the goal channel alone runs at about 52 us/step. Producing the channel and carrying it are separate costs, measured below: the four dims are set algebra over content tokens, and they widen the feature vector from 60 to 64.

| dataset | steps | build us/step v4 -> v5 | bytes/step v4 -> v5 | dims |
|---|---|---|---|---|
| langgraph7b_real2 | 174 | 808 -> 833 | 480 -> 512 | 60 -> 64 |
| autogen7b_real2 | 148 | 802 -> 805 | 480 -> 512 | 60 -> 64 |
| real_research7b_long_drift | 276 | 793 -> 823 | 480 -> 512 | 60 -> 64 |
