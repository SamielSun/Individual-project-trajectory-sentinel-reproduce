# Forensic accounting: every miss and false alarm

**SUPPORTING / diagnostic evidence, not a performance claim.** This is an interior view of a result that has already been reported (`results/framework_generalized_monitor_report.md`), which stands unchanged as the pre-fix baseline. Reading these corpora this way spends them: they are development data from here on.

Every counterfactual arm below is calibrated on the SAME healthy fold at the SAME 5% budget as the shipped monitor, so 'would this have alarmed' is always asked at equal false-alarm cost. `cusum_rate` divides the running CUSUM by the number of scored steps, `max_increment` drops accumulation entirely; both are causal and both peak at a step the monitor could act on.

## langgraph

- calibration fold: 24 healthy episodes, T = 4..9; held-out healthy test 24, injected 24
- shipped `theta` (max_cusum, empirical, 5%): **191.860**
- calibration-fold episode maxima: min 0.21, median 31.40, max 241.33; 0 of 24 are exactly 0.0
- budget-matched thresholds per arm and statistic:

  | arm | max_cusum | cusum_rate | max_increment |
  |---|---|---|---|
  | em | 191.860 | 57.321 | 59.482 |
  | e | 191.860 | 57.321 | 59.482 |
  | m | 18.692 | 12.685 | 13.812 |
  | x | 51.434 | 15.919 | 21.196 |

## autogen

- calibration fold: 24 healthy episodes, T = 2..12; held-out healthy test 24, injected 23
- shipped `theta` (max_cusum, empirical, 5%): **5.974**
- calibration-fold episode maxima: min 0.00, median 0.00, max 33.47; 14 of 24 are exactly 0.0
- budget-matched thresholds per arm and statistic:

  | arm | max_cusum | cusum_rate | max_increment |
  |---|---|---|---|
  | em | 5.974 | 1.630 | 3.734 |
  | e | 5.974 | 1.630 | 3.734 |
  | m | 2.259 | 0.533 | 2.064 |
  | x | 3.017 | 0.377 | 1.668 |

## Every miss and false alarm

| framework | episode | class | T | tau | outcome | max fused | theta | moved dims (post-onset peak, x healthy ceiling) | budget-matched arms that would alarm | root cause |
|---|---|---|---|---|---|---|---|---|---|---|
| langgraph | langgraph-real-tool_cascade-003 | tool_cascade | 7 | 5 | miss | 92.05 | 191.86 | e16=3.6(x1.47);e25=3.35(x1.22);e31=2.19(x1.06) | e:max_increment;em:max_increment;x:max_increment | decision_statistic_loses_signal |
| langgraph | langgraph-real-tool_cascade-002 | tool_cascade | 5 | 2 | miss | 165.98 | 191.86 | e23=3.03(x1.25);e20=2.19(x1.15);e13=2.56(x1.04) | e:cusum_rate;e:max_increment;em:cusum_rate;em:max_increment;x:cusum_rate;x:max_cusum;x:max_increment | decision_statistic_loses_signal |
| langgraph | langgraph-real-tool_cascade-000 | tool_cascade | 4 | 2 | miss | 75.74 | 191.86 | e13=2.66(x1.09);e5=2.24(x1.08);latency_log=3.13(x1.08);outlen_log=2.85(x1.04) | e:cusum_rate;e:max_increment;em:cusum_rate;em:max_increment;m:cusum_rate;m:max_increment;x:cusum_rate;x:max_increment | decision_statistic_loses_signal |
| langgraph | langgraph-real-goal_drift-004 | goal_drift | 7 | 3 | miss | 126.20 | 191.86 | e6=2.96(x1.18) | e:max_increment;em:max_increment;m:max_cusum;m:max_increment;x:max_cusum;x:max_increment | decision_statistic_loses_signal |
| langgraph | langgraph-real-goal_drift-000 | goal_drift | 4 | 2 | miss | 4.75 | 191.86 | (none above ceiling) | (none) | telemetry_lacks_signal |
| langgraph | langgraph-real-goal_drift-003 | goal_drift | 7 | 3 | miss | 0.88 | 191.86 | e21=2.4(x1.01);e19=2.42(x1.01) | (none) | cusum_accumulates_too_slowly |
| langgraph | langgraph-real-goal_drift-005 | goal_drift | 8 | 3 | miss | 180.40 | 191.86 | e21=3.41(x1.44);e25=3.57(x1.3);e17=2.48(x1.19);tool_latency=2.01(x1.0) | e:max_increment;em:max_increment | decision_statistic_loses_signal |
| langgraph | langgraph-real-context_corruption-002 | context_corruption | 5 | 3 | miss | 104.13 | 191.86 | e6=3.01(x1.2);e18=2.21(x1.17);outlen_log=2.87(x1.05);latency_log=3.01(x1.04);self_consistency=1.79(x1.01) | e:max_increment;em:max_increment | decision_statistic_loses_signal |
| langgraph | langgraph-real-goal_drift-001 | goal_drift | 7 | 3 | miss | 0.45 | 191.86 | (none above ceiling) | (none) | telemetry_lacks_signal |
| autogen | autogen-real-healthy-107 | healthy | 7 |  | false_alarm | 46.98 | 5.97 | e5=3.17(x1.21);e23=2.51(x1.09);task_sim=2.26(x1.02) | e:cusum_rate;e:max_cusum;e:max_increment;em:cusum_rate;em:max_increment;x:cusum_rate;x:max_cusum;x:max_increment | decision_statistic_admits_healthy_transient |
| autogen | autogen-real-healthy-098 | healthy | 6 |  | false_alarm | 13.43 | 5.97 | tool_latency=2.52(x1.12);e13=2.78(x1.12);self_consistency=2.52(x1.1);e16=2.8(x1.02);e7=2.82(x1.01);e22=2.16(x1.01) | e:cusum_rate;e:max_cusum;e:max_increment;em:cusum_rate;em:max_increment;m:cusum_rate;m:max_cusum;m:max_increment;x:cusum_rate;x:max_cusum;x:max_increment | healthy_transient_resembles_failure |
| autogen | autogen-real-healthy-091 | healthy | 12 |  | false_alarm | 88.13 | 5.97 | (none above ceiling) | e:cusum_rate;e:max_cusum;e:max_increment;em:cusum_rate;em:max_increment;m:cusum_rate;m:max_cusum;m:max_increment;x:cusum_rate;x:max_cusum;x:max_increment | healthy_transient_resembles_failure |
| autogen | autogen-real-goal_drift-004 | goal_drift | 4 | 2 | miss | 0.00 | 5.97 | e7=3.34(x1.2) | (none) | residual_below_healthy_centre |
| autogen | autogen-real-goal_drift-002 | goal_drift | 4 | 2 | miss | 0.00 | 5.97 | (none above ceiling) | (none) | residual_below_healthy_centre |

## Root-cause counts

- **decision_statistic_loses_signal**: 6
- **telemetry_lacks_signal**: 2
- **healthy_transient_resembles_failure**: 2
- **residual_below_healthy_centre**: 2
- **cusum_accumulates_too_slowly**: 1
- **decision_statistic_admits_healthy_transient**: 1

## Per-class detection on this development corpus

| framework | class | detected / n |
|---|---|---|
| autogen | context_corruption | 6 / 6 |
| autogen | goal_drift | 4 / 6 |
| autogen | looping | 6 / 6 |
| autogen | tool_cascade | 5 / 5 |
| langgraph | context_corruption | 5 / 6 |
| langgraph | goal_drift | 1 / 6 |
| langgraph | looping | 6 / 6 |
| langgraph | tool_cascade | 3 / 6 |
