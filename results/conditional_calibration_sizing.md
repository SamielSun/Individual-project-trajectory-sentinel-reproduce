# Sizing the healthy calibration fold for the conditional detector

The conditional design was held back on the grounds that its second stage is
calibrated on too few episodes: a gate opening on a quarter of healthy runs
leaves only a quarter of the calibration fold for the adoption threshold, and
in the three goal corpora that quarter is 6 to 14 episodes. Before building the
complementary design — a flat detector and a conditional `g` to `c` path
sharing one budget — the question was how many healthy episodes it would take
to calibrate that second stage properly.

The answer is that the second stage is not the expensive part, and sizing it is
not what has been blocking the design. The requirement is the same as the flat
detector's, and the flat detector does not meet it either.

## What governs the precision

Both thresholds are rank statistics. A threshold placed at the `k`-th largest
of `n` healthy calibration values fires on a fresh healthy episode with
probability `Beta(k, n + 1 - k)` — a distribution that depends on `k` and `n`
alone and not at all on how the scores are distributed. Its coefficient of
variation is about `1 / sqrt(k)`. So the precision of a false-alarm calibration
is set by **how many calibration episodes sit above the final threshold**,
not by the size of the fold underneath it.

For the flat detector that count is `0.05 x N`. For the two-stage detector it
is `p2 x (p1 x N)`, and since `p1 x p2 = 0.05` by construction, it is
`0.05 x N` as well — the same number, for every allocation of the budget
between the stages.

| allocation | n_calib | exceedances above final threshold | realized FA, 95th pct | CV |
|---|---|---|---|---|
| flat 0.05 | 320 | 16 | 0.0712 | 0.243 |
| gate 0.25 / cond 0.20 | 320 | 16 | 0.0705 | 0.243 |
| gate 0.50 / cond 0.10 | 320 | 16 | 0.0708 | 0.244 |
| gate 1.00 / cond 0.05 | 320 | 16 | 0.0710 | 0.243 |

The conditional fold is small, but the rate it has to resolve is
correspondingly coarse: a fold of 80 resolving 20% is exactly as precise as a
fold of 320 resolving 5%. Splitting the budget across two stages costs nothing
in calibration precision. The earlier reading — that the conditional stage
needed several times more healthy data than the flat one — was wrong.

## Where the corpora actually stand

| design | n_calib | exceedances | median realized FA | 95th pct | P(realized ≤ 5%) |
|---|---|---|---|---|---|
| langgraph, hierarchy 0.25/0.20 | 24 | 1 | 0.024 | 0.102 | 0.765 |
| autogen, hierarchy 0.25/0.20 | 54 | 2 | 0.029 | 0.079 | 0.794 |
| long_drift, hierarchy 0.25/0.20 | 24 | 1 | 0.024 | 0.102 | 0.766 |
| deployment, flat 0.05 (frozen) | 80 | 4 | 0.046 | 0.094 | 0.572 |

The conditional thresholds in the goal corpora are being set by the single
largest conditional value in the fold, which is not a calibration. The frozen
flat detector is better placed but not by much: its 5% threshold is the 4th
largest of 80, so the rate it actually achieves on fresh healthy episodes has a
95th percentile of 9.4%.

This is a statement about finite-sample calibration, not a relaxation of the
budget. The 5% target stands. What the table shows is how much evidence the
current folds provide that a given deployment meets it — and the honest answer
is that a nominal 5% threshold read off 80 episodes supports a claim of "no
worse than about 10% with 95% confidence", not a claim of 5%.

## What it would take

Calibrating at a nominal 5% puts the median realized rate at 5%, so half of all
deployments overshoot by construction. Holding the budget with confidence means
accepting a bounded overshoot. The fold needed for each tolerance:

| realized FA ≤ tolerance × nominal | exceedances | n_calib | 95th pct |
|---|---|---|---|
| 2.0× | 4 | 80 | 0.095 |
| 1.5× | 13 | 260 | 0.074 |
| 1.3× | 32 | 640 | 0.065 |
| 1.2× | 70 | 1400 | 0.060 |
| 1.1× | 267 | 5340 | 0.055 |

A separate and much weaker constraint is resolution: the conditional threshold
exists at all only when `floor(p2 x p1 x N) >= 1`, i.e. `N >= 20` for any
allocation. Below that the quantile helper falls back to a threshold above
every calibration value, and the realized conditional rate becomes `1/(n+1)`
regardless of the rate that was requested. All three goal corpora clear
resolution and fail precision.

## The collection cost is mostly an allocation choice

Under the split the studies use today — 30% development holdout, then 50% fit,
20% scale, 30% calibration of the remainder — only 21% of a healthy corpus
reaches the calibration fold, so every calibration episode costs about five
collected ones. That proportion is not required. The fit fold estimates a
readout and the scale fold a dispersion; both are adequate at their current
sizes and neither has any reason to grow with the false-alarm budget. Holding
them fixed and giving every further healthy episode to calibration:

| tolerance | n_calib | healthy episodes, proportional folds | healthy episodes, fixed fit + scale |
|---|---|---|---|
| 2.0× | 80 | 381 | 297 |
| 1.5× | 260 | 1239 | 477 |
| 1.3× | 640 | 3048 | 857 |
| 1.2× | 1400 | 6667 | 1617 |

The deployment corpus in hand is 297 healthy episodes split 140 fit / 40 scale
/ 80 threshold / 37 development. Reallocated under a fixed-fold design it gives
the same 80 calibration episodes it already has — the 2.0× row. Reaching 1.5×
needs 477 healthy episodes per deployment, an additional 180 on top of what
exists.

## The model was checked against the implemented rule, not just derived

The Beta-product derivation assumes more than it states: that the two stages
are independent, that the conditional population the second threshold is
calibrated on is exchangeable with the one it meets at serving time, and that
the serving-time passage count behaves like the fixed count the calibration
fold takes on. None of those are free. The gate threshold is estimated from the
same sample that supplies the conditional values, so the conditional fold is
selected by its own gate statistic; if the two channels are correlated within
an episode that selection is not neutral. At serving time passage is random.

So the implemented alarm rule was run end to end against a population where the
truth is computable -- 1,500 calibration draws per configuration, each realized
rate measured on 20,000 fresh healthy episodes -- with a per-episode latent term
driving both channels so the correlation could be swept rather than assumed
away.

| allocation | rho | n_calib | median realized (emp / model) | 95th pct (emp / model) |
|---|---|---|---|---|
| gate 0.25 / cond 0.20 | 0.0 | 80 | 0.0419 / 0.0434 | 0.0888 / 0.0897 |
| gate 0.25 / cond 0.20 | 0.9 | 80 | 0.0440 / 0.0434 | 0.0906 / 0.0899 |
| gate 0.25 / cond 0.20 | 0.9 | 320 | 0.0488 / 0.0483 | 0.0722 / 0.0704 |
| gate 0.50 / cond 0.10 | 0.9 | 320 | 0.0495 / 0.0486 | 0.0730 / 0.0707 |
| gate 1.00 / cond 0.05 | 0.9 | 320 | 0.0500 / 0.0487 | 0.0716 / 0.0710 |

Across all eighteen configurations the largest disagreement in the 95th
percentile is **0.0043**, and it does not grow with correlation: rho = 0.9
tracks rho = 0 as closely as either tracks the model. The analytic sizing can
be relied on for this procedure.

## What an "at most 5%" claim actually costs

Everything above sizes the *average* case: calibrating at a nominal 5% puts the
median realized rate at 5%, so half of deployments land above it. A claim that
the false alarm rate is at most 5% is a different and stronger statement, and
it needs a different rule.

The distribution-free one-sided tolerance bound gives it: choose the k-th
largest of n healthy calibration values, with k the largest integer whose
`Beta(k, n+1-k)` 95th percentile still fits under 5%. That guarantees, with 95%
confidence, that a fresh healthy episode alarms at most 5% of the time -- for
any score distribution, with no parametric assumption. It is paid for in
sensitivity, because the nominal rate it can afford is k/(n+1), strictly below
the budget:

| n_calib | k | nominal rate afforded | guaranteed upper bound | budget usable |
|---|---|---|---|---|
| 80 | 1 | 0.0123 | 0.0368 | 25% |
| 160 | 4 | 0.0248 | 0.0477 | 50% |
| 320 | 10 | 0.0312 | 0.0486 | 62% |
| 640 | 23 | 0.0359 | 0.0487 | 72% |
| 1280 | 51 | 0.0398 | 0.0492 | 80% |
| 3000 | 131 | 0.0437 | 0.0500 | 87% |

At the 80 calibration episodes available today the rule can only afford a
nominal 1.2% -- a threshold set at the single largest healthy value -- which
throws away three quarters of the budget and with it most of the detector's
sensitivity. That is the real cost of the current fold, and it is why 80
episodes reproduce the existing nominal-5% procedure but cannot support the
final claim.

### Predeclared

For the final production standard, the false alarm calibration uses the
**distribution-free 95% one-sided tolerance bound**: threshold at the k-th
largest healthy calibration value, k the largest integer with
`BetaInv(0.95; k, n+1-k) <= 0.05`. The standard fold is **n_calib = 640**,
which affords a nominal 3.6% under a guaranteed ceiling of 4.87% and keeps 72%
of the budget usable. With the fit and scale folds held at their current
adequate sizes that is **857 healthy episodes per deployment**.

This is predeclared rather than chosen after seeing results, and it applies to
the flat detector and the conditional design alike. Two alternatives are on the
same curve if the collection cost is judged wrong: n_calib = 320 keeps 62% of
the budget for 537 episodes, and n_calib = 1280 keeps 80% for 1,497.

## Consequence for the complementary design

The conditional design is not blocked on a conditional-fold problem specific to
it. It is blocked on the same calibration evidence the flat detector is blocked
on, and it should be built when that evidence exists rather than before.

Until then the corrected disjoint fit / scale / calibration procedure stays as
it is, the 5% total budget is unchanged, and the complementary design remains
unrun.

Derivation: `_internal/deployment_healthy_collection/size_conditional_calibration.py`
and `validate_two_stage_sizing.py`; figures in
`conditional_calibration_sizing.json` and `two_stage_sizing_validation.json`.
