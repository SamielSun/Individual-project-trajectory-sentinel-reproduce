# The tool-reliability channel r_t

## Root cause

`error_flag` (dim 42) is per-step and memoryless, so an isolated flake and the
third step of a cascade produce the same feature value. Healthy runs already
carry errors on about 9% of steps, so error *presence* has almost no contrast
with the healthy null — the class was failing on contrast, not sensitivity.
What distinguishes a cascade is the **shape** of failure over the episode: how
many distinct tools are affected, whether the same call fails again, and how
long since anything worked.

## The new dimensions

Seven dims at `[64, 71)`, `telemetry v6`, in `derail/telemetry/reliability.py`.
Every dim is causal (steps 0..t only), bounded in [0, 1], oriented so higher is
more anomalous, and computed only from tool names, canonical argument keys and
error flags.

| dim | name | what it measures |
|---|---|---|
| 64 | fail_frac | share of this step's calls that errored |
| 65 | distinct_failing | distinct tools failing and not yet recovered, over distinct tools used so far |
| 66 | repeat_fail | share of this step's failures whose exact (name, args) already failed — a retry that fails again is not flakiness |
| 67 | consec_fail | consecutive failing calls across steps |
| 68 | since_clean | steps since the last successful tool result |
| 69 | no_recovery | share of tools that failed and never recovered |
| 70 | burst | mean failure fraction over the last 3 steps |

Counts are squashed through `1 - exp(-n/3)` rather than used raw: an integer
count has a lumpy healthy null that a small calibration fold estimates badly,
which is exactly how the earlier run-length feature lost its false-alarm
control. `since_clean` only starts running after the episode's first tool call,
so a purely reasoning prefix does not read as degradation. A trace with no tool
events leaves all seven dims 0 — inert, never an error.

Nothing in the channel sees a label, a future step, a task, a corpus or a
framework. It reads `ToolCallEvent` fields alone, which both the structured
`tool_events` path and the text-parsing fallback populate.

## Dims [0, 64) are unchanged

The block is appended above the goal channel exactly as `c` was appended above
`g`. Re-deriving every relevant corpus with `reliability=True` and comparing
against the same corpus at `goal=True`:

| corpus | width | dims [0,64) identical | episodes with a non-zero r |
|---|---|---|---|
| langgraph7b_real2 | 64 → 71 | **True** (144/144) | 144/144 |
| autogen7b_real3 | 64 → 71 | **True** (144/144) | 144/144 |
| autogen7b_real2 | 64 → 71 | **True** (143/143) | 142/143 |
| real_research7b_long_drift | 64 → 71 | **True** (240/240) | 240/240 |

671 episodes, every element of `[0, 64)` equal. No corpus was recollected —
telemetry is derived from the stored traces, so re-deriving was enough.

## Attempted, unrealized, realized

An injected episode is only *realized* if the failure actually occurred: for
`goal_drift` by the existing realization rule, for the error classes if an
errored tool call appears at or after tau. The injector ramps
probabilistically, so an attempted episode need not contain its own failure.
This split is evaluation-only and never reaches the detector.

| corpus | class | attempted | realized | unrealized |
|---|---|---|---|---|
| langgraph | context_corruption | 6 | 4 | 2 |
| langgraph | goal_drift | 6 | 3 | 3 |
| langgraph | looping | 6 | 6 | 0 |
| langgraph | tool_cascade | 6 | 6 | 0 |
| autogen | context_corruption | 6 | 2 | 4 |
| autogen | goal_drift | 6 | 3 | 3 |
| autogen | looping | 6 | 6 | 0 |
| autogen | tool_cascade | 6 | 6 | 0 |
| long_drift | goal_drift | 120 | 74 | 46 |

No episode lacked post-onset runway.

## Each dim on its own, before fusion

Episode AUC against the healthy null, realized episodes only, post-onset steps
only.

| dim | context (LG) | goal_drift (LG) | looping (LG) | tool_cascade (LG) | tool_cascade (AG) |
|---|---|---|---|---|---|
| fail_frac | 0.915 | 0.581 | 0.915 | 0.831 | 0.850 |
| **distinct_failing** | 0.915 | 0.581 | 0.943 | **0.943** | **0.944** |
| repeat_fail | 0.741 | 0.491 | 0.991 | 0.658 | 0.565 |
| consec_fail | 0.921 | 0.567 | 0.999 | 0.897 | 0.926 |
| since_clean | 0.837 | 0.457 | **1.000** | 0.881 | 0.884 |
| no_recovery | 0.915 | 0.581 | 0.915 | 0.915 | 0.933 |
| burst | 0.938 | 0.558 | 0.991 | 0.892 | 0.924 |

`distinct_failing` — scope — beats the existing `error_flag` (0.831 / 0.850) on
tool cascade, which is the hypothesis confirmed: presence is weak, scope is
strong. `since_clean` and `consec_fail` are near-perfect on looping. On goal
drift every dim sits near 0.5, and on the long-drift corpus at 0.46-0.50: the
channel is specific and adds no signal where tools are not failing, which is
the correct behaviour rather than a shortcoming.

`repeat_fail` is constant on healthy in the long-drift corpus and is correctly
excluded there.

## Detection, per class, causal post-onset

20 replicates, four disjoint healthy folds, same rank rule for every arm.

**langgraph7b_real2**

| arm | FA | detection | context | goal_drift | looping | tool_cascade |
|---|---|---|---|---|---|---|
| e+m/max (frozen) | 0.034 | 0.425 | 0.633 | 0.017 | 0.683 | 0.367 |
| e+m+g/mean | 0.021 | 0.448 | 0.567 | 0.025 | 0.875 | 0.325 |
| **r/mean alone** | 0.044 | 0.479 | 0.292 | 0.000 | 0.992 | **0.633** |
| e+m+g/mean ∪ lvl | 0.030 | 0.569 | **0.658** | 0.325 | **1.000** | 0.292 |
| **e+m+g/mean ∪ r/mean ∪ lvl** | **0.021** | **0.592** | 0.583 | 0.225 | **1.000** | **0.558** |

**autogen7b_real3**

| arm | FA | detection | context | goal_drift | looping | tool_cascade |
|---|---|---|---|---|---|---|
| e+m/max (frozen) | 0.022 | 0.240 | 0.083 | 0.067 | 0.742 | 0.067 |
| e+m+g/mean | 0.022 | 0.260 | 0.017 | 0.100 | 0.817 | 0.108 |
| lvl (r dims included) | 0.087 | 0.396 | 0.142 | 0.217 | 0.958 | 0.267 |
| e+m+g/mean ∪ lvl | 0.018 | 0.323 | 0.075 | 0.133 | 0.917 | 0.167 |

**real_research7b_long_drift** (goal drift only)

| arm | FA | detection realized |
|---|---|---|
| e+m+g/mean (frozen best) | 0.037 | **0.519** |
| e+m+g/mean ∪ lvl | 0.035 | 0.481 |
| e+m+g/mean ∪ r/mean ∪ lvl | 0.047 | 0.339 |

## What improved, what regressed

**tool_cascade on LangGraph: 0.325 → 0.633** with the channel alone, or 0.558
in the union that also keeps looping at 1.000 and context at 0.583, all at
2.1% false alarms. That is the assigned target class, roughly doubled.

**looping improves everywhere** — 0.875 → 0.992 on LangGraph, and 1.000 in the
union arm. No regression on context corruption in the best union arm (0.583 vs
0.567 frozen).

**Regressions.** On the long-drift corpus adding `r` costs realized goal drift
(0.519 → 0.339 in the triple union). The channel is inert there, and a mean
fusion dilutes the channels that do carry signal; the union limits but does not
remove the cost. On AutoGen `r/mean` could not be built at all: the healthy
scale fold has no tool errors, so the channel's healthy null is constant and
`fit_channel_scales` correctly refuses to normalise it.

That refusal is a real limitation of the fusion rather than of the channel. A
dim that is identically zero on healthy runs is the *ideal* detector — any
non-zero value is anomalous — but the normalizer cannot scale it and the
deviation monitor excludes it. Handling a degenerate-healthy dim as a one-sided
indicator instead of dropping it is the obvious next move and is not done here.

## Not at target

The requirement is 100% detection of genuinely realized failures. The best arm
reaches 0.558 on tool cascade for LangGraph and 0.167 for AutoGen. False alarms
are within 5% on every arm reported as a candidate, but two exploratory arms
exceeded it (`lvl` at 0.087 and `all` at 0.176 on AutoGen) and are not
candidates. Nothing here is frozen and no final calibration data was collected.

Channel: `derail/telemetry/reliability.py`. Analysis:
`_internal/deployment_healthy_collection/analyze_reliability_dims.py` and
`evaluate_deviation.py`; figures in `reliability_dim_analysis.json` and
`deviation_arms.json`.
