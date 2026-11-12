# Fusion dilution: what it cost, and a union that does not pay it

## The two reported drops were causal, not resampling

The previous stage reported LangGraph context 0.658 → 0.650 and AutoGen
detection 0.323 → 0.300 as being inside replicate spread and confined to arms
without the reliability branch. **That reading was wrong.** Re-run under common
random numbers — identical corpora, split seed, folds and monitor seeds, with
only the deviation monitor's dim set varied — the `lvl` branch's per-episode
score arrays differ on 750 of 2080 episodes (LangGraph), 344 of 2020 (AutoGen)
and 2553 of 4040 (long drift). `e+m+g/mean`, carried as a control the dim set
cannot touch, is identical throughout.

| arm | context (LG) | detection (AG) |
|---|---|---|
| `e+m+g ∪ lvl`, reliability dims IN the dim set | **0.658** | **0.323** |
| `e+m+g ∪ lvl`, reliability dims OUT | 0.650 | 0.300 |

Both previously reported figures reproduce exactly, and the difference is
wholly explained by reverting `DERIVED_DIMS`. So it was a real regression, and
the trade it implied was real too: the same revert is what recovered long-drift
realized drift from 0.554 to 0.578.

## Where the tool_cascade signal was going

A ladder, one rung per transformation, scored two ways: episode AUC of realized
post-onset maxima against healthy development maxima, and detection at a 5%
threshold from the calibration fold.

| rung | AUC (LG) | det@5% | AUC (AG) | det@5% |
|---|---|---|---|---|
| raw dim `distinct_failing` | 0.929 | 1.000 (FA 0.214) | 0.925 | 0.667 (FA 0.100) |
| after two-sided per-tail scaling | 0.929 | 1.000 | 0.925 | 0.667 |
| top-k over the 7 dims | 0.949 | 0.600 | 0.929 | 0.483 |
| + CUSUM (`rlvl`) | 0.945 | 0.417 | 0.926 | 0.400 |
| ESN channel `r/mean` | 0.946 | 0.633 | 0.892 | 0.217 |

**Nothing is lost in scaling, in the temporal detector, or in fusion.** AUC is
flat to within noise from raw dim to fused channel. Every bit of the loss is at
the operating point: an AUC of 0.93 simply does not give detection near 1.0 at a
5% budget, because healthy runs already carry tool errors on ~9% of steps and
the two distributions overlap precisely in the top 5%. Dims that do reach
detection 1.000 (`fail_frac`, `no_recovery`) do it at 16-21% false alarms.

Reaching the 100% target on this class therefore needs more separation, not
better fusion. One thing fusion *can* fix is visible here: `rlvl` beats
`r/mean` on AutoGen (0.400 vs 0.217), so the reliability dims deserve a branch
of their own rather than a place in someone else's dim set.

## The mechanism

`derail/monitor/evidence_fusion.py`. Two levels, and one property.

**Branch evidence.** Each branch's causal statistic is replaced by its
upper-tail rank in that branch's own healthy null, in -log10 units:
`-log10((count+1)/(n+1))` where `count` is how many healthy scale-fold values
are at least this large, within a relative tie tolerance. A branch at its
healthy floor scores exactly 0; a branch beyond everything the fold saw reaches
`log10(n+1)`.

Above the null maximum the rank saturates, and that matters more than it
sounds: a 16-episode scale fold of ~7 scoreable steps is only ~100 values, so
healthy episodes reach the ceiling routinely, and a rank rule cannot split an
atom — every tied episode alarms at once. The tail is continued past the
ceiling by `log10(1 + excess/beta)`, where beta is the mean excess of the null's
own upper tail, capped at one further ceiling's worth. Two earlier attempts
failed here and both are recorded in the module: counting exceedances from the
95th percentile left 5-6 of them at n≈100 and collapsed every branch to a flat
ceiling (false alarms 9-19%); a *linear* continuation under a hard cap merely
moved the atom from the ceiling to the cap (6-10%).

**Combination.** Evidence is non-negative, so under `max` (Tippett), `sum`
(Fisher) and `top2` alike a branch pinned at its healthy floor contributes
exactly 0 and leaves the combined statistic pointwise unchanged — hence the
same threshold, the same false alarms and the same detections. That is the
invariant an average violates, and `tests/test_evidence_fusion.py` asserts it
for every rule with the averaging control alongside, failing by construction.

**Omnibus.** Which rule is right depends on the failure, not the corpus: a tool
cascade lights one branch hard, where `max` reads it and `sum` averages it
away; goal drift lifts several branches a little, where `sum` accumulates it
and `max` sees nothing decisive. Measured on controlled synthetic scores, power
against a concentrated alternative is max 0.798 / sum 0.704, and against a
diffuse one max 0.322 / sum 0.527. Choosing per corpus would be choosing by
benchmark identity, so instead all three are computed, each is calibrated
against its own healthy null exactly as a branch is, and the maximum is taken —
0.779 and 0.464, never the rule that is wrong. The second level inherits the
non-dilution property because it sees identical streams.

Nothing in the module imports anything but numpy: no label, onset, class,
framework or corpus name can reach it.

## The averaging was the main thing destroying signal

Un-averaging the channels — one branch each instead of `e+m+g/mean` — is worth
more than every other change here combined.

| corpus / metric | `e+m+g/mean` | the single channel alone |
|---|---|---|
| long drift, realized drift | 0.519 | **`g` 0.928** (FA 0.042) |
| LangGraph, context | 0.567 | **`e` 0.758** (FA 0.044) |
| LangGraph, detection | 0.448 | `e` 0.556 |
| AutoGen, detection | 0.260 | **`e` 0.456** (FA 0.034) |
| AutoGen, realized drift | 0.183 | `e` 0.500 |

`fusion_protocol.py` predicted exactly this in its own decision record — "a
fixed budget split across one more channel is the whole effect" — and deferred
the goal channel for that reason. The deferral was the right call for an
average and is the wrong call for a union.

## Results, against every previous best

20 replicates, four disjoint healthy folds, same rank rule for every arm,
detection counted only at or after onset. Attempted/realized/unrealized are
unchanged: LangGraph 6/4/2 context, 6/3/3 drift, 6/6/0 looping, 6/6/0 cascade;
AutoGen 6/2/4, 6/3/3, 6/6/0, 6/6/0; long drift 120/74/46.

Every previous best is now matched or beaten by an in-budget arm:

| corpus | metric | previous best | now | by |
|---|---|---|---|---|
| LangGraph | detection | 0.592 | **0.608** | `sum \| e+m+g+r` |
| LangGraph | realized drift | 0.800 | 0.800 | `lvl` |
| LangGraph | context | 0.658 | **0.758** | `e` |
| LangGraph | looping | 1.000 | 1.000 | several |
| LangGraph | tool_cascade | 0.633 | **0.742** | `sum \| e+m+g+r+rlvl` |
| AutoGen | detection | 0.323 | **0.456** | `e` |
| AutoGen | realized drift | 0.350 | **0.500** | `e` |
| AutoGen | context | 0.083 | **0.375** | `e` |
| AutoGen | looping | 0.917 | **1.000** | `top2 \| e+g+r+rlvl+lvl+all` |
| AutoGen | tool_cascade | 0.183 | **0.283** | `sum \| e+g+r+rlvl+lvl+all` |
| long drift | realized drift | 0.578 | **0.928** | `g` |
| long drift | detection | 0.514 | **0.920** | `g` |

False alarms are inside 5% on every arm listed. The earlier over-budget `max`
arms (0.09-0.19) were the tie artefacts described above and are gone: after the
tail fix no arm in the refined set exceeds 0.05 except the single-branch
`lvl`/`all`/`rlvl` diagnostics on AutoGen, which are not candidates.

## What is NOT finished

**No single arm clears the whole envelope.** The table above is an oracle over
arms — `g` alone wins long drift, `lvl` alone wins LangGraph drift, `e` alone
wins AutoGen context — and a deployment cannot know which. The previous bests
were an envelope over arms too, so this is not a regression, but it is not the
result either.

The binding tension is on LangGraph, and it is specific: the `r` branch is what
takes tool_cascade from 0.325 to 0.742, and it is also what takes realized
drift from 0.500 to 0.000-0.30, because LangGraph healthy runs carry tool
errors and `r`'s healthy noise lifts the union threshold past where `lvl`'s
drift evidence sits. Whether that is irreducible has not been established.

The omnibus rule was measured on LangGraph only before this checkpoint
(`omnibus | e+m+g+lvl+all`: FA 0.031, detection 0.533, realized drift 0.500,
context 0.608, looping 1.000, cascade 0.275) and behaves as designed — near the
better rule rather than the worse one. **AutoGen and long drift were not
reached.** That run is the next thing to do.

Calibration folds here are 27-29 episodes, so `k` is 1 and the threshold is a
single order statistic: coarse, and far below the predeclared 640. These rates
are nominal, no tolerance-bound rank qualifies, nothing is frozen and no final
calibration or evaluation data was collected.

Mechanism: `derail/monitor/evidence_fusion.py`, properties pinned in
`tests/test_evidence_fusion.py`. Analysis:
`_internal/deployment_healthy_collection/prove_regression.py`,
`signal_loss_ladder.py`, `evaluate_fusion{,2,3}.py`, `diagnose_union.py`,
`audit_nothing_worse.py`, `audit_single_arm.py`.
