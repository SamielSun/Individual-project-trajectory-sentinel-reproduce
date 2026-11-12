# The complementary detector: flat `e+m` OR conditional `g` to `c`

The conditional design measured earlier replaced the flat detector and lost
every class the flat detector serves. The design that follows from that is to
run both and take the union — keep `e+m` for the classes it already detects and
add the `g` gate to `c` path beside it. This measures it.

## How the union is calibrated

The two branches are put on a common scale from the scale fold, so the OR
becomes a maximum and the AND a minimum:

```
flat[t]  = fused(e, m)[t] / s_f
hier[t]  = min( runmax(g)[t] / s_g , c[t] / s_c )
union[t] = max( flat[t] , hier[t] )
```

A single threshold is then set on `union` by rank, on a calibration fold
disjoint from fit, scale and development. That threshold is the only quantity
carrying the budget. The three reference rates `s_f`, `s_g`, `s_c` set how
sensitive each branch is relative to the other; they are not budgets and are
read on the scale fold, never on calibration.

This matters because the branch rates do not compose in any fixed way, which
the measurement below shows directly. Four disjoint healthy folds per
replicate, 40 replicates.

## The branch budgets neither add nor multiply

False alarm rates on the development fold, measured at the union threshold:

| corpus | flat branch | hier branch | sum | union measured |
|---|---|---|---|---|
| langgraph7b_real2 | 0.020 | 0.009 | 0.029 | **0.029** |
| autogen7b_real3 | 0.042 | 0.025 | 0.067 | **0.050** |
| real_research7b_long_drift | 0.025 | 0.011 | 0.036 | **0.035** |

On two corpora the branches alarm on disjoint healthy episodes and the rates
add exactly. On AutoGen they overlap by about a quarter and the union comes in
well under the sum. Neither the additive nor the multiplicative assumption
would have been safe, in either direction: allocating by addition would have
left AutoGen spending less than its budget, and any multiplicative allocation
would have overspent on the other two. Calibrating the union itself is the only
thing that holds.

## The union redistributes detection rather than adding it

All three arms share folds, replicates and the same 5% rank rule, thresholded
on their own statistic.

**langgraph7b_real2** (n_calib 29)

| arm | FA | detection | drift realized | context | looping | tool_cascade |
|---|---|---|---|---|---|---|
| flat only | 0.030 | **0.503** | 0.017 | **0.637** | **0.908** | **0.458** |
| hier only | 0.025 | 0.239 | **0.442** | 0.300 | 0.367 | 0.067 |
| union | 0.029 | 0.458 | 0.233 | 0.562 | 0.817 | 0.338 |

**autogen7b_real3** (n_calib 27)

| arm | FA | detection | drift realized | context | looping | tool_cascade |
|---|---|---|---|---|---|---|
| flat only | 0.051 | **0.355** | 0.392 | **0.221** | **0.800** | **0.167** |
| hier only | 0.041 | 0.194 | 0.617 | 0.142 | 0.246 | 0.033 |
| union | 0.050 | 0.326 | **0.650** | 0.204 | 0.608 | 0.138 |

**real_research7b_long_drift** (n_calib 29, goal drift only)

| arm | FA | detection | drift realized |
|---|---|---|---|
| flat only | 0.032 | 0.032 | 0.033 |
| hier only | 0.026 | 0.246 | **0.376** |
| union | 0.035 | 0.184 | 0.271 |

The union is never the best arm on overall detection, and on the long-drift
corpus it is beaten by the conditional path alone: adding a flat branch that
detects 3% of those episodes only forces the shared threshold up. What the
union does buy is realized goal drift where the flat detector is blind —
0.017 to 0.233 on LangGraph, 0.392 to 0.650 on AutoGen — and it pays for that
on looping (0.908 to 0.817, 0.800 to 0.608) and tool cascade (0.458 to 0.338).

The gate-to-adopt ratio matters and the flat share does not: `0.05/0.25/0.20`
and `0.10/0.25/0.20` are identical because the union threshold renormalises
whatever the flat share was, while widening the gate to `0.50/0.10` costs drift
on every corpus.

## Reading

The complementary design works as a mechanism — the union calibrates cleanly,
holds its budget, and does give the flat detector a goal-drift capability it
does not otherwise have. It is not a solution to goal drift. Realized drift
reaches 0.65 at best and 0.23 on LangGraph, against a 100%-per-class target,
and every point of it is bought from classes that were already being detected.
Goal drift stays **OPEN**, and so does `tool_cascade`.

## What this evidence cannot support

The calibration folds here are 27 to 29 healthy episodes. Under the predeclared
final rule — the distribution-free 95% one-sided tolerance bound — **no rank
qualifies at that size**: the smallest fold that admits even `k = 1` is 80, and
the predeclared standard is 640. The thresholds above are the nominal 5% rank
rule, which at `k = 1` is the maximum of the calibration fold, and that is why
the measured false alarm rates land at 3-5% rather than at 5%.

This is development evidence about the design's behaviour. It is not, and
cannot be turned into, evidence that any of these arms meets the frozen
requirement. That needs the predeclared fold.

Study: `_internal/deployment_healthy_collection/union_detector.py`; figures in
`union_detector.json`.
