# The long-corpus goal-drift misses: what they actually are

The framework corpora showed that detection reaches 100% once post-onset
runway exceeds five steps. `real_research7b_long_drift` has that runway in 119
of 120 injected episodes and still detects about a third, so this is the one
population where the detector had every opportunity and failed. This is the
mechanism behind those misses.

Scored under the `e+m+g/mean` arm across twenty independent healthy fold
draws, so a miss is a property of the episode rather than of one threshold.

## There is no separable population of hard episodes

| population | n |
| --- | --- |
| detected in >= 80% of draws | 3 |
| detected in <= 20% of draws | 39 |
| in between | 77 |

Not one episode of 119 is detected in every draw or missed in every draw. The
per-episode detection rate is p25 0.20, median 0.25, p75 0.43. **Whether an
episode is caught is decided by where theta lands, not by the episode.**

The reason is that the whole injected population sits just under the
threshold. Post-onset peak as a fraction of theta: p25 0.58, median 0.72,
p75 0.90, p95 1.64. **96 of 119 episodes never reach theta at all.** The
signal is present and pointing the right way; it is uniformly too small.

## Nothing about the trajectory separates caught from missed

Comparing the stable extremes -- detected in >= 80% of draws against detected
in <= 20%:

| metric | detected | missed |
| --- | --- | --- |
| T | 12.00 | 10.97 |
| tau | 2.00 | 2.00 |
| post-onset runway | 9.00 | 7.97 |
| tool calls | 10.00 | 9.97 |
| distinct tools | 4.00 | 4.00 |
| tool calls after onset | 8.00 | 7.97 |
| output tokens after onset | 30.63 | 30.46 |
| text length after onset | 194.4 | 199.7 |

Every trajectory-shape measure is the same to within noise. Goal evolution,
action trajectory, tool vocabulary and output volume do not distinguish them.
One quantity does.

## What distinguishes them is whether the failure happened at all

| realization | detected >= 80% | detected <= 20% |
| --- | --- | --- |
| `answer_realized` | 1.000 | 0.385 |
| `action_realized` | 0.000 | 0.026 |
| `not_realized` | 0.000 | **0.590** |

**Fifty-nine per cent of the stably-missed episodes are `not_realized`** -- the
injector fired, and the agent ignored it. Those are not failures. Detecting
them would be a false alarm carrying an injected label, which is exactly what
the standing constraint that attempted injection is not realized failure
exists to prevent.

Against the honest denominator, restricted to runway > 5:

| population | episodes | detection |
| --- | --- | --- |
| `answer_realized` | 70 | 0.354 |
| `action_realized` | 3 | 0.233 |
| `not_realized` | 46 | **0.288** |
| realized only | 73 | 0.349 |

## The mechanism: the detector fires on arrival, the label records adoption

The detector alarms on 28.8% of hijacks the agent ignored against 35.4% of
those it adopted. That gap is far too small for a detector that is supposed to
be finding adopted drift, and it identifies what the alarm is actually
responding to: the arrival of an ungrounded payload, not the agent's use of
it. Those are different events and the corpus labels the second.

Per-channel, on injected episodes at runway > 5:

| channel | separates adopted from ignored | separates injected from healthy |
| --- | --- | --- |
| `e` | 0.556 | 0.881 |
| `m` | 0.607 | 0.524 |
| `g` | 0.580 | **0.980** |
| `c` | **0.890** | 0.739 |

`g` is close to perfect at seeing that an injection landed and close to blind
to whether it was adopted. `c` is the reverse. The fused detector is currently
carrying the channel that answers the first question while being scored
against a label that asks the second.

**A caveat that must not be skipped:** `c` and the realization rule
operationalise the same construct -- the agent's output being sourced from
tool output rather than from the goal -- by different routes, one label-free
and one using the known payload. `c`'s 0.890 is therefore substantially
construct overlap, not independent validation, and it is **not** grounds for
reopening the decision to reject `c`. That rejection rests on what `c` costs
other classes under a shared false-alarm budget, which was measured
separately and is unaffected by this.

## Conclusion: not an observability blocker, a budget and target mismatch

The information needed to separate adopted drift from ignored drift exists in
the telemetry. What does not exist is room for it: a single scalar statistic
under a single 5% false-alarm budget is being asked to answer two questions
with different base rates, and adding the channel that answers the second
costs about thirty points on every other failure class.

So goal drift stays OPEN, and it is not blocked on new telemetry. It is
blocked on the detector having one output where the problem has two questions.
Recording this as the mechanism; not proposing the architecture that follows
from it until that is asked for.

## Evaluation methodology, changed as a result

Detection is now reported stratified by post-onset runway (`<=2`, `3-5`, `>5`)
in `results/deployment_detector_freeze.json`, alongside the pooled figure.
Two rules follow:

- A miss at runway `<=2` is runway-limited: the episode ended before enough
  post-onset evidence existed. It stays counted as a miss and is reported
  separately, so it is never read as detector accuracy.
- LangGraph and AutoGen detection must not be compared without controlling for
  runway. A step is not the same quantity in the two frameworks -- LangGraph
  emits reasoning as separate scoreable events, so it yields more steps for
  the same agent work -- and within runway bands the two frameworks detect
  identically.
