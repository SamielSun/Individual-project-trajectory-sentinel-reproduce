# One threshold per branch, not one threshold per union

## The defect, stated exactly

Under the `max` rule an episode's score is the largest evidence any branch
reaches, and the operating point is one order statistic of the healthy
calibration fold. `floor(0.05 * 29)` is 1, so theta is the single largest
healthy calibration episode, and therefore a maximum over episodes AND
branches:

    theta(B u {j}) == max( theta(B), extreme(j) )

`diagnose_r_floor.py` checks both sides per replicate on LangGraph. The
identity holds exactly in 8 of 8.

The consequence is not statistical, it is arithmetic. Adding a branch never
lowers theta, and a branch contributes nothing to a failure it cannot read, so
every failure episode whose peak falls in `[theta(B), extreme(j))` is lost the
moment branch `j` joins. In replicate 4, `theta` without `r` was 1.703 and
`r`'s own healthy extreme was 2.501; three realized-drift episodes sat at 2.17,
2.08 and 2.11, all carried by `lvl`, and all three were lost. In the other
seven replicates `r`'s extreme fell below the incumbent theta and cost nothing.

That is the reliability channel's "noise floor" in precise terms. `r` is not
noisy in the usual sense -- it is the LEAST bursty branch measured
(`diagnose_r_conflict.py`: mean episode-max evidence 0.500 below the
exchangeable-steps expectation, against 0.239 for `m`), and it sets the
threshold in only one replicate of eight. It acts as a floor, not a tail.

The same identity indicts the whole union, not just `r`. `lvl` alone reaches
LangGraph realized drift 0.800; the five-branch union without `r` already drops
it to 0.383. A failure that lights one branch must out-score the LOUDEST
branch's healthy extreme rather than its own.

## The fix

Give each branch its own cut, taken from its own healthy calibration episodes,
and alarm on the first step at which any branch clears its own cut. Written as
a margin,

    S(t) = max_j ( evidence_j(t) - c_j ),   c_j = max over healthy calib of
                                                  branch j's episode evidence

thresholded just above zero, so every existing scoring path applies unchanged.
`c_lvl` now depends only on `lvl`'s healthy episodes, and no branch's healthy
extreme can move another's bar. The identity above is broken by construction.

Nothing here reads a corpus name, a framework, a label, an onset or an outcome.
The cuts are healthy-development order statistics and the branch set is fixed
in advance and identical on every corpus.

## What it costs, and the law that governs it

A cut at the maximum of `n` healthy episodes is exceeded by a fresh healthy
episode with probability about `1/(n+1)`, so a `k`-branch union costs roughly

    FA(n, k) ~ c * k / (n + 1)

`calib_size_law.py` measures it by subsampling the calibration fold. On
LangGraph, `c ~ 0.66`; positive dependence between branches makes the union
cheaper than the Bonferroni bound, but the linear growth in `k` and the decay
in `n` both hold, and the same shape reproduces on AutoGen.

This is why branch-local thresholds were rejected earlier in this work and why
that rejection was wrong. They were only ever tried on the six- and
seven-branch unions, where `k/(n+1)` is 0.093 against a 0.05 budget. The
failure was of the WIDTH, not of the mechanism.

## Mechanisms measured and rejected

| mechanism | verdict | evidence |
|---|---|---|
| per-branch cuts at `alpha/k` (Bonferroni, Sidak) | rejected | `alpha/k` = 0.0073 is finer than the dev fold resolves (1/35); realised per-branch rates summed to 0.093 (`diagnose_branch_cut.py`) |
| branch-local standardisation, calib fold | rejected | FA 0.076-0.105; loc/scale and threshold from the same fold |
| branch-local standardisation, scale fold | rejected | in budget, but at a MATCHED false-alarm rate it equals the global cut (`ORACLE-*` rows); the apparent gain was an operating-point move |
| persistence windows, w = 2, 3 | rejected | realized drift 0.833 -> 0.556 -> 0.111 on `e+g+lvl`; the failure evidence is itself few-step, so smoothing destroys it |
| combination rules `sum`, `top2`, `omnibus` | no resolution | every per-metric best rises, no arm clears the envelope; the same tension persists |
| POT-smoothed global threshold | partial | genuine variance reduction (`e+m+g+r+lvl+all`: detection 0.550 -> 0.604, drift 0.283 -> 0.433 at comparable FA), but it does not reach the corner |
| **per-branch cuts on a minimal union** | **works** | clears every previous best on every corpus and class; blocked only by the budget |

## The tradeoff frontier was an artifact of the shared threshold

Under the global cut the tension is real and measured. `pareto_frontier.py`
takes every arm inside the 5% budget on all three corpora -- 42 of them -- and
finds 6 on the LangGraph realized-drift vs tool-cascade frontier. It is a clean
monotone trade, cascade 0.250 -> 0.675 as drift falls 0.750 -> 0.083, and no
arm reaches the target corner of (0.800, 0.633).

Under per-branch cuts the corner is reached. `PB|r+lvl` gives realized drift
0.800 and tool cascade 0.642 in one arm, at LangGraph FA 0.036.

So the frontier was a property of the shared threshold, not of the signals.

## Causality and the realization split

`account_arm.py` re-derives every detection from raw crossing indices rather
than trusting the scorer. For `PB|r+lvl` over 20 replicates:

* **pre-onset crossings excluded as non-causal: 0** on all three corpora. No
  detection in any number here is a crossing that preceded the injection step.
* detection latency after onset: median 1 step on both framework corpora,
  median 6 on long drift, maximum 10.
* the realization split behaves as it must. On LangGraph the monitor detects
  `action_realized` 16/20 and `answer_realized` 32/40, both 0.800, and
  `not_realized` **0/60 = 0.000**. A hijack that never took hold produces no
  alarm, which is why `drift_realized` excludes that stratum rather than
  counting it as a miss.

## What the mechanism reaches, and what still blocks it

At the frozen split, `PB|e+g+r+lvl` -- one fixed branch set, per-branch cuts,
identical on every corpus -- clears **all thirteen** corpus/class bars at once:

| corpus | metric | previous best | `PB e+g+r+lvl` |
|---|---|---|---|
| LangGraph | detection | 0.592 | **0.754** |
| LangGraph | realized drift | 0.800 | **0.833** |
| LangGraph | context corruption | 0.658 | **0.842** |
| LangGraph | looping | 1.000 | 1.000 |
| LangGraph | tool cascade | 0.633 | **0.758** |
| AutoGen | detection | 0.323 | **0.554** |
| AutoGen | realized drift | 0.350 | **0.683** |
| AutoGen | context corruption | 0.083 | **0.417** |
| AutoGen | looping | 0.917 | **1.000** |
| AutoGen | tool cascade | 0.183 | **0.425** |
| long drift | detection | 0.514 | **0.916** |
| long drift | realized drift | 0.578 | **0.930** |
| long drift | goal drift | 0.514 | **0.916** |

It is **not deployable**: the worst-corpus false-alarm rate is 0.136 against a
0.05 budget. That is the only unmet constraint, and it is the `k/(n+1)` law
with `k = 4`.

## The blocker is calibration volume, and this measures how much

`calib_headroom.py` re-allocates the fold fractions to trade dev episodes for
calibration episodes, holding `fit` and `scale` at their current sizes, so the
monitor being thresholded is unchanged and only the threshold's sample grows.

| corpus | arm | calib 29 | calib ~40 |
|---|---|---|---|
| LangGraph | `PB e+r+lvl` | FA **0.066** | FA **0.043** |
| AutoGen | `PB e+r+lvl` | FA 0.060 | FA **0.034** |
| long drift | `PB e+g+r+lvl` | FA 0.136 | FA 0.085 |

At `calib = 40`, `PB|e+r+lvl` is inside the budget on **both** framework
corpora and still clears every bar on both -- LangGraph detection 0.729,
realized drift 0.833, context 0.767, looping 1.000, cascade 0.733 at FA 0.043;
AutoGen 0.435 / 0.417 / 0.275 / 1.000 / 0.250 at FA 0.034. Ten of ten, one
fixed arm, inside budget.

Long drift needs `g`, so it needs `k = 4`, and at `calib = 42` it is still at
0.085. Solving `c k/(n+1) = 0.05` with its measured `c ~ 0.91` puts the
requirement near **70 healthy calibration episodes**.

Caveat: the re-allocated arm buys calibration episodes with dev episodes, so
its false-alarm estimate is read off 22-23 episodes and is correspondingly
coarse. It establishes the direction and the order of magnitude, not a
certified rate.

## Where this leaves the protocol

The tension the previous checkpoint recorded as possibly irreducible is not
irreducible. It was a property of sharing one threshold across branches, and
per-branch cuts remove it. What replaces it is a sizing requirement: a
four-branch detector at a 5% budget needs on the order of 70 healthy
calibration episodes, and the frozen split yields 27-29.

That is a statement about how much healthy data to collect, not about fusion,
and the planned healthy collection supplies it several times over.
## 40 common-random-number replicates

`confirm40.py`, same seeds and folds, reps 0-19 reproducing the 20-replicate
pass. One AutoGen replicate is dropped for every arm alike: on that fold the
goal channel had no estimable scale and `channel_streams` refused to build it,
so 39 of 40 are carried there.

**The conflict this work set out to resolve is resolved, inside the budget.**
`PB|r+lvl` on LangGraph, FA **0.046**:

| metric | previous best | `PB r+lvl` @40 |
|---|---|---|
| detection | 0.592 | **0.624** |
| realized drift | 0.800 | **0.800** |
| looping | 1.000 | **1.000** |
| tool cascade | 0.633 | **0.633** |
| context corruption | 0.658 | 0.621 |

`r` no longer costs realized drift. Under the global cut the same two branches
gave either drift 0.800 with cascade 0.242, or cascade 0.642 with drift 0.000.

At 40 replicates `PB|e+g+r+lvl` still clears **all thirteen** bars -- LangGraph
0.748 / 0.825 / 0.833 / 1.000 / 0.746, AutoGen 0.535 / 0.590 / 0.380 / 1.000 /
0.415, long drift 0.902 / 0.913 / 0.902 -- and is still out of budget, at FA
0.091 / 0.075 / 0.113.

### A caveat about the bars themselves

At 40 replicates the incumbent `OLD e+m+g/meanUlvl` does not reproduce the
LangGraph bars it set: detection 0.549 against 0.592, realized drift 0.708
against 0.800. Those bars are an envelope recorded over 20-replicate passes,
and they are optimistic. Judged like-for-like in this run against the same
baseline on the same folds, `PB|e+g+r+lvl` wins 12 of 13 cells and ties the
13th -- nothing worse anywhere -- while spending three times the false alarms.

## Remaining regressions and unmet targets

1. **The budget.** `PB|e+g+r+lvl` sits at FA 0.075-0.113 against 0.05. This is
   the only constraint it fails, and `calib_headroom.py` shows it closing with
   calibration volume rather than with any change of mechanism.
2. **No single in-budget arm covers all three corpora.** `PB|r+lvl` takes
   LangGraph 4/5 but reads nothing on long drift (0/3); `GLOB|e+g+lvl` is in
   budget everywhere and takes AutoGen 4/5 and long drift 3/3 but LangGraph
   0/5. Covering all three needs `e`, `g`, `r` and `lvl` together, which is
   `k = 4`, which is the sizing requirement above.
3. **Two envelope cells are not met by any in-budget arm at 40 replicates**:
   LangGraph context corruption (best 0.621 against 0.658) and AutoGen looping
   (best 0.889 against 0.917).

## Status of the gates

`py -m pytest -m "not network and not ollama"` -- 885 passed, 1 skipped (the
pre-existing `deployment freeze not built yet`).
`behavior_snapshot`, `claims_ledger`, `data_card`, `baseline_lock` -- all pass.
`artifact_manifest --check` fails, and did so before this work: `CHECKSUMS.md`
was last regenerated at `90ca7772`, six commits before `2b57c499`, so
`deviation.py` and `evidence_fusion.py` read as added. Left alone; regenerating
it is a release action.

Calibration folds here are 27-29 episodes, `k` is 1 and the threshold is a
single order statistic. These rates are nominal, no tolerance-bound rank
qualifies, nothing is frozen, and no final calibration or evaluation data was
collected or consulted.

Mechanism: `evaluate_perbranch.py`. Diagnosis: `diagnose_r_conflict.py`,
`diagnose_r_floor.py`, `diagnose_branch_cut.py`, `calib_size_law.py`,
`calib_headroom.py`. Rejected mechanisms: `evaluate_branch_local.py`,
`evaluate_persistence.py`, `evaluate_final_rules.py`. Analysis:
`pareto_frontier.py`, `account_arm.py`, `show_arm.py`, `confirm40.py`.
