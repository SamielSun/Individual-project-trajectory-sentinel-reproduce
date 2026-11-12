# Why goal drift and tool cascade were missed, and what closes part of the gap

Both open classes were being missed for the same reason, and it was not a
missing signal. Scoring the raw 64-dim telemetry directly — no reservoir, no
fusion, per-dimension Mann-Whitney AUC of post-onset steps against the healthy
null — shows the information is present and in several cases nearly perfect.
The frozen monitor cannot use it because it only ever looks up.

## The signal is there, in the lower tail

Two-sided separability, `max(AUC, 1 - AUC)`, on realized episodes:

| corpus / class | dimension | two-sided AUC | direction |
|---|---|---|---|
| long_drift / goal_drift | grd_char_anom | **1.000** | LOW |
| long_drift / goal_drift | grd_self_dis | **1.000** | LOW |
| long_drift / goal_drift | grd_consec_dis | 0.998 | LOW |
| long_drift / goal_drift | grd_drift | 0.996 | LOW |
| langgraph / goal_drift | goal_arg_abandon | **1.000** | high |
| langgraph / goal_drift | goal_arg_echo | 0.991 | high |
| langgraph / goal_drift | grd_query_dis | 0.937 | LOW |
| autogen / goal_drift | goal_arg_abandon | 0.997 | high |
| langgraph / tool_cascade | error_flag | 0.831 | high |
| langgraph / tool_cascade | grd_query_dis | 0.802 | LOW |
| autogen / tool_cascade | outlen_log | 0.730 | LOW |
| autogen / tool_cascade | latency_log | 0.722 | LOW |

An AUC of 0.000 is as informative as 1.000 — it separates perfectly, in the
other direction. The mechanism is straightforward once seen. After a goal
hijack the agent locks onto one narrow topic, so successive tool results become
*more* alike and steps get shorter; under a tool cascade, short formulaic error
strings replace documents, so latency and output length *drop*. A one-sided
statistic reads both as healthier than healthy.

## Three changes, each general and label-free

**Two-sided scoring.** Deviation from the healthy null in either direction is
anomalous. The null is estimated on healthy fit episodes alone.

**A separate scale per tail.** Scaling by one spread statistic lets the fat
side of a skewed dim set the divisor and hides excursions into the thin side.
This was not cosmetic: with a shared scale the long-drift corpus, whose
grounding dims separate perfectly downward, still scored as healthy. Each tail
is scaled by the deviation at which that side alone would spend 5% of the
healthy budget, which also puts every dim on one footing regardless of units.

**A causal persistence feature.** How many consecutive steps a dim has stayed
on one side of its healthy centre. Tool cascade is a contrast problem rather
than a sensitivity one -- healthy runs already carry errors on 8.8% of steps,
so an error is not by itself unusual. What is unusual is an error that does not
clear: isolated flakiness gives a run of one, a cascade gives a run that keeps
growing.

**A windowed spread feature.** On the long-drift corpus post-onset steps are
not extreme at all — they are unusually uniform. No per-step deviation can see
a collapse in variability, so the windowed standard deviation of each dim is
scored as a feature in its own right.

Selecting dims by their tail probability rather than by a spread statistic
matters too: a raw IQR z-score lets the heaviest-tailed dim win the maximum
regardless of how discriminative it is. An earlier attempt using the empirical
tail probability directly saturated — with 28 dims almost every episode has one
dim outside the healthy range, so scores tied at the floor and the detector
alarmed on everything (false alarm rate 1.000).

## What it buys

20 replicates, four disjoint healthy folds, same rank rule for every arm,
detection counted only at or after onset.

**langgraph7b_real2**

| arm | FA | detection | drift realized | context | looping | tool_cascade |
|---|---|---|---|---|---|---|
| e+m+g/mean (frozen) | 0.021 | 0.448 | 0.050 | 0.567 | 0.875 | **0.325** |
| deviation, level | 0.030 | 0.502 | **0.800** | 0.467 | **1.000** | 0.142 |
| **e+m+g/mean + deviation** | 0.024 | **0.569** | 0.750 | **0.650** | **1.000** | 0.250 |

**autogen7b_real3**

| arm | FA | detection | drift realized | context | looping | tool_cascade |
|---|---|---|---|---|---|---|
| e+m+g/mean (frozen) | 0.022 | 0.260 | 0.183 | 0.017 | 0.817 | 0.108 |
| deviation, level | 0.081 | 0.383 | **0.467** | 0.158 | 0.875 | **0.242** |
| **e+m+g/mean + deviation** | 0.018 | **0.300** | 0.350 | **0.083** | 0.817 | 0.125 |

**real_research7b_long_drift**

| arm | FA | detection | drift realized |
|---|---|---|---|
| e+m+g/mean (frozen) | 0.037 | **0.511** | 0.519 |
| deviation, level+spread | 0.040 | 0.210 | 0.334 |
| e+m+g/mean + deviation (level) | 0.037 | 0.433 | 0.468 |
| e+m+g/mean + deviation (level+spread+persist) | 0.043 | 0.514 | **0.578** |

The branches are combined the way the union study established as the only sound
method: each divided by its own scale-fold tail value, then a maximum, then one
threshold on the combined statistic from the calibration fold. Only that
threshold carries the budget.

Realized goal drift on LangGraph goes from 0.050 to 0.750 — the frozen detector
was finding one realized hijack in twenty, the combined detector three in four.
False alarms stay inside budget on every corpus and arm reported here.

## What it does not buy

**No arm dominates everywhere.** The level feature carries the framework
corpora; spread and persistence carry long drift; the fused channels carry
context corruption. `e+m+g/mean + level` is best on both framework corpora and
loses 0.519 to 0.468 on long drift; adding spread and persistence recovers long
drift to 0.578, above the frozen detector, but costs LangGraph. Choosing one
configuration for all three is currently a trade, not a win.

**tool_cascade is barely moved.** The persistence feature gives the best figure
yet on LangGraph — 0.400 against 0.325 for the frozen detector and 0.367 for
`e+m/max` — but at a false alarm rate of 0.063, outside budget, and the same
feature is unusable on AutoGen, where a CUSUM over run lengths reaches a false
alarm rate of 0.268. Run lengths are integer-valued and their healthy null is
poorly estimated on a 30-episode fit fold, which is the likely cause. Against a
target of 1.000 the class is not close.

The remaining gap is a telemetry one. `error_flag` is per-step and memoryless,
and persistence recovers only part of what is missing: a cascade is
distinguished from flakiness by its SCOPE and its failure to recover — how many
distinct tools are affected, whether a retry of the same call also fails, how
long recovery takes. None of that is in the current 64 dims. It does not need
new collection: telemetry is derived from the stored traces, so new dims can be
computed by re-deriving the existing corpora.

**None of this is at the target.** No class reaches 1.000 on every corpus, and
the standing requirement is 100% observed detection per class. This is progress
on a stated blocker, not closure.

## Scope of the evidence

Arms were selected on these development corpora, which is legitimate for
development and cannot support a final claim. Calibration folds here are 27-35
episodes, far below the predeclared 640, so the tolerance-bound rule does not
apply and these false alarm rates are nominal-rank estimates. Nothing was
frozen.

Diagnosis: `_internal/deployment_healthy_collection/diagnose_open_classes.py`.
Monitor: `derail/monitor/deviation.py`, with properties pinned in
`tests/test_deviation_monitor.py`. Evaluation:
`_internal/deployment_healthy_collection/evaluate_deviation.py`; figures in
`open_class_diagnosis.json` and `deviation_arms.json`.
