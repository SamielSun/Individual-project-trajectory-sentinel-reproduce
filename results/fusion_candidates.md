# Fusing on a common healthy-null scale

**SUPPORTING / diagnostic evidence, development data only.** These corpora were spent when their interior was opened; they choose a design here and the design is frozen before anything fresh is collected. No number below is a performance claim, and the pre-fix baseline (`results/framework_generalized_monitor_report.md`) stands unchanged.

Healthy episodes are admitted by the frozen rule: the task's own verifier must pass. Unscoreable verified-healthy episodes are never used to fit, scale or threshold, and never enter a threshold distribution as zeros; they are counted in the operational false-alarm denominator, because a deployment still runs them.

`scoreable FA` divides by the healthy episodes the detector could act on; `operational FA` divides by every verified-healthy episode. They differ exactly by the monitoring coverage.

| framework | arm | protocol | fit / scale / threshold | theta | detection (95% CI) | scoreable FA (95% CI) | operational FA | AUC | p50 us/step |
|---|---|---|---|---|---|---|---|---|---|
| langgraph | cusum/raw | holdout | 49 / 16 / 17 | 56.264 | 0.458 (0.256-0.672) | 0.057 (0.007-0.192) | 0.057 | 0.839 | 116.6 |
| langgraph | cusum/raw | crossfit | 62 / 20 / 82 | 79.141 | 0.458 (0.256-0.672) | 0.000 (0.000-0.100) | 0.000 | 0.848 | 116.0 |
| langgraph | cusum/normalized | holdout | 49 / 16 / 17 | 6.048 | 0.542 (0.328-0.744) | 0.057 (0.007-0.192) | 0.057 | 0.845 | 119.1 |
| langgraph | cusum/normalized | crossfit | 62 / 20 / 82 | 17.904 | 0.875 (0.676-0.973) | 0.400 (0.239-0.579) | 0.400 | 0.868 | 118.0 |
| langgraph | rate/normalized | holdout | 49 / 16 / 17 | 8.262 | 0.375 (0.188-0.594) | 0.000 (0.000-0.100) | 0.000 | 0.839 | 115.6 |
| langgraph | rate/normalized | crossfit | 62 / 20 / 82 | 13.174 | 0.875 (0.676-0.973) | 0.429 (0.263-0.606) | 0.429 | 0.861 | 116.3 |
| langgraph | increment/normalized | holdout | 49 / 16 / 17 | 12.992 | 0.292 (0.126-0.511) | 0.000 (0.000-0.100) | 0.000 | 0.815 | 115.9 |
| langgraph | increment/normalized | crossfit | 62 / 20 / 82 | 11.485 | 0.708 (0.489-0.874) | 0.029 (0.001-0.149) | 0.029 | 0.884 | 120.7 |
| autogen | cusum/raw | holdout | 40 / 13 / 14 | 45.503 | 0.696 (0.471-0.868) | 0.103 (0.022-0.274) | 0.086 | 0.858 | 115.0 |
| autogen | cusum/raw | crossfit | 50 / 17 / 67 | 71.372 | 0.609 (0.385-0.803) | 0.034 (0.001-0.178) | 0.029 | 0.889 | 119.0 |
| autogen | cusum/normalized | holdout | 40 / 13 / 14 | 24.669 | 0.696 (0.471-0.868) | 0.103 (0.022-0.274) | 0.086 | 0.864 | 117.1 |
| autogen | cusum/normalized | crossfit | 50 / 17 / 67 | 455.546 | 0.217 (0.075-0.437) | 0.000 (0.000-0.119) | 0.000 | 0.897 | 113.7 |
| autogen | rate/normalized | holdout | 40 / 13 / 14 | 89.625 | 0.609 (0.385-0.803) | 0.034 (0.001-0.178) | 0.029 | 0.885 | 115.6 |
| autogen | rate/normalized | crossfit | 50 / 17 / 67 | 786.052 | 0.000 (0.000-0.148) | 0.000 (0.000-0.119) | 0.000 | 0.909 | 118.3 |
| autogen | increment/normalized | holdout | 40 / 13 / 14 | 11.135 | 0.826 (0.612-0.950) | 0.069 (0.008-0.228) | 0.057 | 0.892 | 118.2 |
| autogen | increment/normalized | crossfit | 50 / 17 / 67 | 20.066 | 0.826 (0.612-0.950) | 0.069 (0.008-0.228) | 0.057 | 0.897 | 113.9 |

## Arm comparison over 40 dev-splits

A single split of this pool leaves a threshold fold of 17-20 episodes, which the committed sample-size curve puts at the edge of identifiability, so one split cannot separate the arms. Every arm sees the same splits and differences are paired against the shipped `cusum/raw`; Wilcoxon is over splits. The native corpora cannot supply more data here: they record `success = None`, so the frozen admission rule admits none of them as verified healthy.

| framework | arm | detection (sd) | block means | d vs shipped | Wilcoxon p | Holm p | scoreable FA | d vs shipped | Holm p | m win share |
|---|---|---|---|---|---|---|---|---|---|---|
| autogen | cusum/normalized | 0.425 (0.177) | 0.357;0.493 | +0.008 | 0.556 | 1.000 | 0.047 | -0.001 | 1.000 | 0.248 |
| autogen | cusum/raw | 0.417 (0.158) | 0.372;0.463 | -- | -- | -- | 0.047 | -- | -- | 0.261 |
| autogen | increment/normalized | 0.510 (0.223) | 0.450;0.570 | +0.092 | 0.000 | 0.004 | 0.039 | -0.009 | 1.000 | 0.314 |
| autogen | rate/normalized | 0.538 (0.198) | 0.522;0.554 | +0.121 | 0.004 | 0.033 | 0.041 | -0.007 | 1.000 | 0.257 |
| langgraph | cusum/normalized | 0.521 (0.150) | 0.529;0.512 | +0.009 | 0.968 | 1.000 | 0.056 | +0.000 | 1.000 | 0.369 |
| langgraph | cusum/raw | 0.511 (0.142) | 0.527;0.496 | -- | -- | -- | 0.056 | -- | -- | 0.341 |
| langgraph | increment/normalized | 0.501 (0.156) | 0.535;0.467 | -0.010 | 0.984 | 1.000 | 0.051 | -0.004 | 1.000 | 0.444 |
| langgraph | rate/normalized | 0.539 (0.142) | 0.573;0.504 | +0.027 | 0.199 | 1.000 | 0.055 | -0.001 | 1.000 | 0.377 |

## Which channel wins the fusion

The defect this addresses: on the shipped design the semantic channel wins almost every step because its CUSUM is simply larger, not because it is better informed.

| framework | arm | protocol | e wins | m wins |
|---|---|---|---|---|
| langgraph | cusum/raw | holdout | 0.621 | 0.379 |
| langgraph | cusum/raw | crossfit | 0.573 | 0.427 |
| langgraph | cusum/normalized | holdout | 0.405 | 0.595 |
| langgraph | cusum/normalized | crossfit | 0.922 | 0.078 |
| langgraph | rate/normalized | holdout | 0.332 | 0.668 |
| langgraph | rate/normalized | crossfit | 0.922 | 0.078 |
| langgraph | increment/normalized | holdout | 0.422 | 0.578 |
| langgraph | increment/normalized | crossfit | 0.750 | 0.250 |
| autogen | cusum/raw | holdout | 0.826 | 0.174 |
| autogen | cusum/raw | crossfit | 0.764 | 0.236 |
| autogen | cusum/normalized | holdout | 0.876 | 0.124 |
| autogen | cusum/normalized | crossfit | 0.893 | 0.107 |
| autogen | rate/normalized | holdout | 0.848 | 0.152 |
| autogen | rate/normalized | crossfit | 0.904 | 0.096 |
| autogen | increment/normalized | holdout | 0.837 | 0.163 |
| autogen | increment/normalized | crossfit | 0.685 | 0.315 |

## Length strata

A statistic that merely relocated the length bias would look fine in the aggregate. Detection and false alarms are reported per trajectory-length band, with the median alarm step.

| framework | arm | protocol | band | n healthy | FA | n injected | detection | median alarm step |
|---|---|---|---|---|---|---|---|---|
| langgraph | cusum/raw | holdout | 4-5 | 11 | 0.000 | 4 | 0.000 | nan |
| langgraph | cusum/raw | holdout | 6-7 | 20 | 0.050 | 10 | 0.200 | 5.5 |
| langgraph | cusum/raw | holdout | 8-+ | 4 | 0.250 | 10 | 0.900 | 5.0 |
| langgraph | cusum/raw | crossfit | 4-5 | 11 | 0.000 | 4 | 0.000 | nan |
| langgraph | cusum/raw | crossfit | 6-7 | 20 | 0.000 | 10 | 0.200 | 6.0 |
| langgraph | cusum/raw | crossfit | 8-+ | 4 | 0.000 | 10 | 0.900 | 6.0 |
| langgraph | cusum/normalized | holdout | 4-5 | 11 | 0.000 | 4 | 0.000 | nan |
| langgraph | cusum/normalized | holdout | 6-7 | 20 | 0.050 | 10 | 0.400 | 5.0 |
| langgraph | cusum/normalized | holdout | 8-+ | 4 | 0.250 | 10 | 0.900 | 5.0 |
| langgraph | cusum/normalized | crossfit | 4-5 | 11 | 0.455 | 4 | 0.750 | 3.0 |
| langgraph | cusum/normalized | crossfit | 6-7 | 20 | 0.300 | 10 | 0.800 | 3.0 |
| langgraph | cusum/normalized | crossfit | 8-+ | 4 | 0.750 | 10 | 1.000 | 4.0 |
| langgraph | rate/normalized | holdout | 4-5 | 11 | 0.000 | 4 | 0.250 | 3.0 |
| langgraph | rate/normalized | holdout | 6-7 | 20 | 0.000 | 10 | 0.100 | 5.0 |
| langgraph | rate/normalized | holdout | 8-+ | 4 | 0.000 | 10 | 0.700 | 5.0 |
| langgraph | rate/normalized | crossfit | 4-5 | 11 | 0.545 | 4 | 0.750 | 3.0 |
| langgraph | rate/normalized | crossfit | 6-7 | 20 | 0.300 | 10 | 0.800 | 3.0 |
| langgraph | rate/normalized | crossfit | 8-+ | 4 | 0.750 | 10 | 1.000 | 4.0 |
| langgraph | increment/normalized | holdout | 4-5 | 11 | 0.000 | 4 | 0.000 | nan |
| langgraph | increment/normalized | holdout | 6-7 | 20 | 0.000 | 10 | 0.000 | nan |
| langgraph | increment/normalized | holdout | 8-+ | 4 | 0.000 | 10 | 0.700 | 6.0 |
| langgraph | increment/normalized | crossfit | 4-5 | 11 | 0.000 | 4 | 0.250 | 4.0 |
| langgraph | increment/normalized | crossfit | 6-7 | 20 | 0.050 | 10 | 0.600 | 5.0 |
| langgraph | increment/normalized | crossfit | 8-+ | 4 | 0.000 | 10 | 1.000 | 4.0 |
| autogen | cusum/raw | holdout | 4-5 | 16 | 0.062 | 10 | 0.400 | 4.0 |
| autogen | cusum/raw | holdout | 6-7 | 8 | 0.125 | 5 | 0.800 | 4.0 |
| autogen | cusum/raw | holdout | 8-+ | 5 | 0.200 | 8 | 1.000 | 4.0 |
| autogen | cusum/raw | crossfit | 4-5 | 16 | 0.000 | 10 | 0.200 | 3.5 |
| autogen | cusum/raw | crossfit | 6-7 | 8 | 0.000 | 5 | 0.800 | 5.0 |
| autogen | cusum/raw | crossfit | 8-+ | 5 | 0.200 | 8 | 1.000 | 4.0 |
| autogen | cusum/normalized | holdout | 4-5 | 16 | 0.062 | 10 | 0.400 | 4.0 |
| autogen | cusum/normalized | holdout | 6-7 | 8 | 0.125 | 5 | 0.800 | 4.0 |
| autogen | cusum/normalized | holdout | 8-+ | 5 | 0.200 | 8 | 1.000 | 4.0 |
| autogen | cusum/normalized | crossfit | 4-5 | 16 | 0.000 | 10 | 0.000 | nan |
| autogen | cusum/normalized | crossfit | 6-7 | 8 | 0.000 | 5 | 0.000 | nan |
| autogen | cusum/normalized | crossfit | 8-+ | 5 | 0.000 | 8 | 0.625 | 7.0 |
| autogen | rate/normalized | holdout | 4-5 | 16 | 0.000 | 10 | 0.600 | 3.0 |
| autogen | rate/normalized | holdout | 6-7 | 8 | 0.000 | 5 | 0.400 | 3.0 |
| autogen | rate/normalized | holdout | 8-+ | 5 | 0.200 | 8 | 0.750 | 3.5 |
| autogen | rate/normalized | crossfit | 4-5 | 16 | 0.000 | 10 | 0.000 | nan |
| autogen | rate/normalized | crossfit | 6-7 | 8 | 0.000 | 5 | 0.000 | nan |
| autogen | rate/normalized | crossfit | 8-+ | 5 | 0.000 | 8 | 0.000 | nan |
| autogen | increment/normalized | holdout | 4-5 | 16 | 0.000 | 10 | 0.600 | 3.0 |
| autogen | increment/normalized | holdout | 6-7 | 8 | 0.125 | 5 | 1.000 | 4.0 |
| autogen | increment/normalized | holdout | 8-+ | 5 | 0.200 | 8 | 1.000 | 4.0 |
| autogen | increment/normalized | crossfit | 4-5 | 16 | 0.000 | 10 | 0.600 | 3.0 |
| autogen | increment/normalized | crossfit | 6-7 | 8 | 0.125 | 5 | 1.000 | 4.0 |
| autogen | increment/normalized | crossfit | 8-+ | 5 | 0.200 | 8 | 1.000 | 4.0 |

## The protocol this selects

Recorded as a checked artifact in `results/fusion_protocol_decision.json` and pinned by a test, so the choice cannot drift from the code that implements it.

- statistic **rate**: the only arm positive on both deployments, significant on one after Holm correction, and the one that flattens the false-alarm gradient across trajectory-length bands;
- per-channel normalization **on**, but not because it detects more -- it does not, on either deployment. It is what stops the fusion being decided by which channel carries larger numbers;
- threshold on the **fused** statistic at a 5% budget, never inferred from per-channel budgets;
- **three_fold_holdout**: fit >= 140, scale >= 40, threshold >= 80, untouched test 300 per deployment. Cross-fitting was measured and rejected.

## Healthy score against prefix length

Mean fused score on held-out healthy episodes by how many steps have been scored. A statistic that has decoupled length should be flat here; one that has not will climb.

| framework | arm | prefix steps | mean score | n |
|---|---|---|---|---|
