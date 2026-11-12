# The served detector, frozen

`normalized_fusion[e+m]/rate/v2`, frozen on the deployment's own healthy population.

The pre-fix detector in `results/framework_monitor_freeze.json` is untouched
and stays the immutable baseline. This is a separate artifact with its own
version, so the published 0.625 / 0.000 and 0.913 / 0.125 numbers remain
independently reproducible from the freeze they were produced by.

## Folds

Four disjoint healthy folds from the 297 verified, scoreable episodes of
`real_task_rollback_healthy`. Admission is the task's own predeclared
verifier, decided at collection time (`results/deployment_healthy_collection_audit.md`).

| fold | n | what it sets |
| --- | --- | --- |
| fit | 140 | standardizer, reservoirs, readouts |
| scale | 40 | each channel's healthy null |
| threshold | 80 | theta, on the fused statistic |
| development | 37 | nothing -- reporting only |

No episode appears in two folds, and the development fold sets no parameter.

## Result

theta = 66.4650, picked on the fused statistic at a 5.0% budget.

| quantity | value | 95% CI |
| --- | --- | --- |
| healthy false alarms, development fold | 2/37 = 5.4% | 1.5% - 17.7% |
| detection, served injected corpus | 14/16 = 87.5% | 64.0% - 96.5% |
| served healthy, transfer check | 3/4 | -- |

### Per class

| class | detected | 95% CI | median steps after onset |
| --- | --- | --- | --- |
| `context_corruption` | 4/4 | 51.0% - 100.0% | 0 |
| `looping` | 4/4 | 51.0% - 100.0% | 1 |
| `tool_cascade` | 2/4 | 15.0% - 85.0% | 1 |
| `wrong_document` | 4/4 | 51.0% - 100.0% | 0 |

Detection counts an alarm only at or after the injected onset, never before it.

## What these numbers are not

The development fold is 37 episodes. At that size an observed
5.4% carries a 95% interval reaching 17.7%, so this
does not establish the <=5% false-alarm target -- it is a development
measurement on a fold that set no parameter, and the target needs the fresh
held-out live population that does not exist yet. The same applies to
detection: 16 injected episodes, four per class.

## `tool_cascade` is weak and is not being closed

`tool_cascade` detects 2/4 where the other three classes are 4/4.
That is a knowingly weak class with a plausible general fix outstanding, so
it is recorded as open. No final evaluation population should be collected
while it stands.

## The goal-consistency channel: still undecided, for a new reason

The previous blocker was fold size -- on a 17-episode threshold fold the
order-statistic floor 1/(n+1) = 0.056 exceeds the 5% budget, so a single
episode set theta and the fused budget could not be divided. That blocker is
gone: the threshold fold here is 80, where theta is the 4th largest.

Measured on identical folds, the two arms are indistinguishable:

| | `e+m` | `e+m+c` |
| --- | --- | --- |
| theta | 66.4650 | 66.4650 |
| development false alarms | 2/37 | 2/37 |
| detection | 14/16 | 14/16 |

Every per-class count is identical, so nothing is lost by adding the channel.
Nor is the channel inert: it wins
50 of 801 scored steps (6.2%) and moves the
episode maximum on 17 of 133 episodes, by up to
9.06. It simply never moves one across theta:
0 episodes change their alarm decision.

**The reason this population cannot decide the channel is that it contains
no goal drift at all.** The deployment's four injected classes are
`tool_cascade`, `looping`, `context_corruption` and `wrong_document`;
`goal_drift` episodes available here: 0. A channel built to detect goal
drift, evaluated where no goal drift occurs, can only show what it costs --
and what it costs is nothing.

So the channel is neither adopted nor discarded. What it needs is an
adequately sized healthy population *for a corpus that carries goal drift*,
which is a property of the next collection, not of this freeze. It stays
a live candidate with a measured cost of zero on this deployment.
