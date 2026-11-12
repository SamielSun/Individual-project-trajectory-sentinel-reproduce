# Fusion combiner and channel set: the development decision

Two questions asked together, because they interact: how the per-channel
statistics are combined (`max` or `mean` of the normalized values), and which
channels enter the combination. Every arm is scored on identical folds within
a replicate, so a paired difference removes the fold draw as a nuisance axis.

Populations. The deployment corpus `real_task_rollback_healthy` (297 verified
healthy, 40 draws) carries four injected classes and **no goal drift at all**,
so it cannot decide a goal-drift channel. The three already-spent development
corpora that do carry goal drift answer that question, at 40 replicates each.

Significance is Wilcoxon over replicates, Holm-corrected within each family.

## Deployment corpus, 40 draws

| arm | detection | healthy FA | pre-onset | context_corr | looping | tool_cascade | wrong_doc |
|---|---|---|---|---|---|---|---|
| `e+m/max` | 0.911 | 0.054 | 0.058 | 0.900 | 1.000 | 0.762 | 0.981 |
| `e+m/mean` | 0.928 | 0.049 | 0.050 | 0.925 | 1.000 | 0.800 | 0.988 |
| `e+m+c/mean` | 0.903 | 0.049 | 0.050 | 0.850 | 1.000 | 0.800 | 0.963 |
| `e+m+g/mean` | 0.927 | 0.047 | 0.037 | 0.875 | 1.000 | 0.838 | 0.994 |
| `e+m+g+c/mean` | 0.914 | 0.048 | 0.042 | 0.844 | 1.000 | 0.819 | 0.994 |

Pre-onset alarms are injected episodes whose FIRST alarm precedes onset. They
are counted as misses in `detection`; the column exists so the mechanism stays
visible rather than being hidden inside the miss count.

### Trajectory-length strata (deployment)

| arm | T<=5 FA / det | T6-8 FA / det | T>8 FA / det |
|---|---|---|---|
| `e+m/max` | 0.033 / 0.944 | 0.062 / 0.765 | -- / 0.996 |
| `e+m/mean` | 0.031 / 0.969 | 0.055 / 0.800 | -- / 0.996 |
| `e+m+c/mean` | 0.021 / 0.912 | 0.066 / 0.785 | -- / 0.982 |
| `e+m+g/mean` | 0.021 / 0.919 | 0.061 / 0.840 | -- / 0.993 |
| `e+m+g+c/mean` | 0.019 / 0.894 | 0.067 / 0.820 | -- / 0.993 |

No healthy development episode exceeds 8 steps, so the T>8 false-alarm cell
has no denominator. It is left empty rather than reported as zero.

### Channel contribution (share of scored steps where the channel is the argmax)

| arm | e | m | g | c |
|---|---|---|---|---|
| `e+m/max` | 0.653 | 0.347 | -- | -- |
| `e+m/mean` | 0.653 | 0.347 | -- | -- |
| `e+m+c/mean` | 0.637 | 0.335 | -- | 0.028 |
| `e+m+g/mean` | 0.564 | 0.304 | 0.132 | -- |
| `e+m+g+c/mean` | 0.562 | 0.303 | 0.131 | 0.004 |

### Block stability, first 20 draws vs all 40

| arm | FA 20 -> 40 | detection 20 -> 40 |
|---|---|---|
| `e+m/max` | 0.057 -> 0.054 | 0.906 -> 0.911 |
| `e+m/mean` | 0.045 -> 0.049 | 0.919 -> 0.928 |
| `e+m+c/mean` | 0.047 -> 0.049 | 0.900 -> 0.903 |
| `e+m+g/mean` | 0.042 -> 0.047 | 0.922 -> 0.927 |
| `e+m+g+c/mean` | 0.041 -> 0.048 | 0.906 -> 0.914 |

## Cross-corpus, the populations that carry goal drift

`drift_inj` is detection over every injected goal_drift episode; `drift_real`
is over those the agent actually adopted. The injector's own records show some
hijacks are ignored, and an alarm on an ignored hijack is a false alarm
wearing an injected label, so both denominators are reported.

### `langgraph7b_real2`

| arm | detection | FA | drift_inj | drift_real | context_corruption | looping | tool_cascade | wrong_document | goal_drift |
|---|---|---|---|---|---|---|---|---|---|
| `e+m/max` | 0.531 | 0.056 | 0.042 | 0.083 | 0.708 | 0.925 | 0.450 | -- | 0.042 |
| `e+m/mean` | 0.553 | 0.059 | 0.033 | 0.067 | 0.717 | 0.967 | 0.496 | -- | 0.033 |
| `e+m+c/mean` | 0.403 | 0.045 | 0.287 | 0.575 | 0.454 | 0.658 | 0.212 | -- | 0.287 |
| `e+m+g/mean` | 0.581 | 0.059 | 0.058 | 0.117 | 0.696 | 0.992 | 0.579 | -- | 0.058 |
| `e+m+g+c/mean` | 0.460 | 0.053 | 0.250 | 0.500 | 0.512 | 0.783 | 0.296 | -- | 0.250 |

### `autogen7b_real2`

| arm | detection | FA | drift_inj | drift_real | context_corruption | looping | tool_cascade | wrong_document | goal_drift |
|---|---|---|---|---|---|---|---|---|---|
| `e+m/max` | 0.541 | 0.066 | 0.287 | 0.475 | 0.575 | 0.933 | 0.335 | -- | 0.287 |
| `e+m/mean` | 0.549 | 0.068 | 0.325 | 0.567 | 0.525 | 0.938 | 0.380 | -- | 0.325 |
| `e+m+c/mean` | 0.385 | 0.072 | 0.446 | 0.817 | 0.246 | 0.671 | 0.135 | -- | 0.446 |
| `e+m+g/mean` | 0.517 | 0.066 | 0.233 | 0.400 | 0.483 | 0.925 | 0.410 | -- | 0.233 |
| `e+m+g+c/mean` | 0.398 | 0.067 | 0.425 | 0.767 | 0.233 | 0.713 | 0.185 | -- | 0.425 |

### `real_research7b_long_drift`

| arm | detection | FA | drift_inj | drift_real | goal_drift |
|---|---|---|---|---|---|
| `e+m/max` | 0.068 | 0.071 | 0.068 | 0.068 | 0.068 |
| `e+m/mean` | 0.069 | 0.069 | 0.069 | 0.070 | 0.069 |
| `e+m+c/mean` | 0.160 | 0.074 | 0.160 | 0.234 | 0.160 |
| `e+m+g/mean` | 0.504 | 0.074 | 0.504 | 0.524 | 0.504 |
| `e+m+g+c/mean` | 0.288 | 0.079 | 0.288 | 0.361 | 0.288 |
## Paired effects that survive Holm correction

Only differences that reach family-wise significance are listed; everything
else moved less than the fold draw does.

| population | comparison vs `e+m/max` | effect | up/down/tie | Holm p |
|---|---|---|---|---|
| deployment | `e+m+g/mean` pre-onset alarms | **-0.0203** | 0/13/27 | 0.0075 |
| deployment | `e+m+g/mean` tool_cascade | **+0.0750** | 14/1/25 | 0.0273 |
| deployment | `e+m/mean` detection | **+0.0172** | 10/0/30 | 0.0348 |
| langgraph | `e+m+g/mean` tool_cascade | **+0.1292** | 24/4/12 | 0.0046 |
| langgraph | `e+m+g/mean` detection | **+0.0500** | 25/8/7 | 0.0273 |
| long_drift | `e+m+g/mean` drift_realized | **+0.4561** | 37/0/3 | 0.0000 |
| long_drift | `e+m+g/mean` detection | **+0.4356** | 37/0/3 | 0.0000 |
| langgraph | `e+m+c/*` context_corruption | -0.30 to -0.33 | 0/19-23/… | 0.0001 |
| langgraph | `e+m+c/*` looping | -0.26 to -0.28 | 0/17/23 | 0.0008 |
| langgraph | `e+m+c/*` tool_cascade | -0.20 to -0.21 | 1-3/20-21/… | 0.0006 |
| autogen | `e+m+c/mean` detection | **-0.1565** | 10/25/5 | 0.0103 |
| autogen | `e+m+g+c/mean` detection | **-0.1435** | 10/26/4 | 0.0098 |
| autogen | `e+m+g+c/mean` context_corruption | **-0.3417** | 1/26/13 | 0.0000 |
| both | `e+m+c/*` and `e+m+g+c/*` drift_realized | +0.29 to +0.42 | — | <0.005 |

## Classification

**`e+m+g/mean` — the leading contender.** Significant gains on three of the
four populations: `tool_cascade` on the deployment and on langgraph, aggregate
detection on langgraph, fewer pre-onset alarms, and realized goal-drift
detection on the long corpus rising 0.068 -> 0.524. **No significant
regression anywhere.** The reservation is autogen, where every movement is
non-significant but the signs are unfavourable (context_corruption -0.092,
realized drift -0.075); that is one corpus and within draw noise, but it is
the thing to watch rather than to explain away.

**`e+m/mean` — a safe, small contender.** Significantly better aggregate
detection on the deployment, never worse on any axis or population. It carries
no goal-drift benefit.

**`c` — REJECTED as a production channel.** It buys goal drift and pays for it
everywhere else, consistently and significantly on both multi-class corpora:
realized drift +0.29 to +0.42, against context_corruption -0.30 to -0.34,
looping -0.22 to -0.28, tool_cascade -0.15 to -0.21, aggregate detection
-0.14 to -0.16. A fixed false-alarm budget divided across one more channel is
the mechanism, and it is reallocation, not improvement. It also does not solve
the problem it targets: its best realized-drift detection on the long corpus
is 0.234. Retained as a measured development candidate; not adopted.

**`e+m+g+c/mean` — DOMINATED, rejected.** It inherits `c`'s regressions
(autogen detection -0.1435, context_corruption -0.3417, both significant) and
is beaten by `e+m+g/mean` on every deployment axis. Run to completion so the
rejection rests on evidence rather than inference.

## What is NOT settled

Adopting `e+m+g/mean` would not close the detector. Known weaknesses remain
and a general improvement is still plausible, so this is a development
baseline, not a final freeze:

- `tool_cascade` is 0.838 on the deployment, 0.579 on langgraph, 0.410 on
  autogen. Well short of the target.
- Realized goal-drift detection is 0.524 on the long corpus, 0.400 on autogen,
  0.117 on langgraph. Goal drift stays OPEN.
- The autogen non-significant regressions need a corpus-level explanation
  rather than a shrug.
