# Salvaging a channel whose healthy null is constant

## The limitation

`fit_channel_scales` maps each channel's causal stream through the location and
dispersion of that channel's own healthy null. A channel with no dispersion has
no divisor, and dividing by the `1e-9` floor would amplify it by a billion and
hand it every fusion — the exact defect normalization exists to prevent. So
such a channel was excluded, and if every channel was constant the fit was
refused outright.

That refusal blocked AutoGen from using the reliability channel at all: its
healthy scale fold contains no tool errors, so every `r` dim is identically
zero and the channel statistic never moves. But a dim that is zero on healthy
runs and positive under failure is not uninformative — it is the *ideal*
one-sided indicator. Excluding it discards the strongest available evidence
precisely where it is cleanest.

## The mechanism, at fit time and at score time

**At fit time.** Dispersion is measured as before. If it exceeds
`_MIN_DISPERSION` (1e-9 — a tolerance, never exact equality) nothing changes;
the channel takes the ordinary `scaled` mode and behaves identically to before.

If it does not, the channel is degenerate and one question is asked: does the
channel's *own telemetry definition* declare that its dims are bounded and that
higher means more anomalous? That declaration lives in
`MONOTONE_BOUNDED_CHANNELS` and currently names `c` and `r`, both of which
state "in [0, 1] … HIGHER = more anomalous" in their channel documentation. `g`
declares the orientation but no bound, so it has no admissible full scale and
stays excluded rather than being given an invented one.

For a declared channel the scale becomes `one_sided` with two numbers:

- **baseline** — the maximum of the observed healthy stream, so every healthy
  step in the scale fold scores exactly zero by construction.
- **full_scale** — the statistic's response to *one* maximally anomalous step.
  It is measured by driving a synthetic probe's channel dims to the top of the
  declared range for a single step, against the same probe held at the declared
  floor, and taking the difference. Configuration only: no corpus, no label, no
  outcome. The probe runs on the channel's own sub-monitor and resets it
  afterwards, so it leaves no state behind.

**At score time.** For a value `v`:

```
excess = v - baseline
if excess <= 1e-6 * full_scale:   0.0
else:            CAP * (1 - exp(-excess / full_scale))
```

Healthy-constant values stay null. Anything below the baseline is null — the
declared direction is respected and no direction is invented. Tiny benign
variation cannot score, because the deadband is relative to the declared scale
rather than to the (absent) observed spread. And the transform saturates at
`CAP = 8.0`, so a degenerate channel can win a fusion but cannot annihilate it;
an unbounded map would recreate the divide-by-a-floor defect in a new form.

`usable` still returns False for a `scaled` channel with no dispersion, so the
old exclusion path is intact for anything not declared.

**Choosing the full scale matters.** The first implementation used a *sustained*
saturation run and produced `full_scale = 1212` on AutoGen — so far beyond
anything a real episode reaches that genuine departures mapped to a sliver of
the cap and the channel could never win a fusion. The one-step response gives
228, which puts the channel on the same order as the scaled channels
(healthy mean 0.630, p95 3.985 against looping 4.730).

## AutoGen r is unlocked

| corpus | mode | full_scale | built? |
|---|---|---|---|
| autogen7b_real3 | **one_sided** | 228.4 | **yes — previously refused** |
| langgraph7b_real2 | scaled (sd) | — | yes, unchanged |
| real_research7b_long_drift | scaled (iqr) | — | yes, unchanged |

`r/mean` on AutoGen now scores: healthy dev mean 0.630, looping 4.730,
tool_cascade 2.811, goal_drift 0.514 — inert on drift, as intended.

## Results, against the previous best

20 replicates, four disjoint healthy folds, causal post-onset detection.
Attempted / realized / unrealized are unchanged from the reliability-channel
report: langgraph 6/4/2 context, 6/3/3 drift, 6/6/0 looping, 6/6/0 cascade;
autogen 6/2/4, 6/3/3, 6/6/0, 6/6/0; long_drift 120/74/46.

**autogen7b_real3 — the target**

| arm | FA | detection | context | goal_drift | looping | tool_cascade |
|---|---|---|---|---|---|---|
| e+m/max | 0.022 | 0.240 | 0.083 | 0.067 | 0.742 | 0.067 |
| e+m+g/mean | 0.022 | 0.260 | 0.017 | 0.100 | 0.817 | 0.108 |
| **r/mean (newly buildable)** | 0.028 | 0.279 | 0.033 | 0.000 | 0.900 | **0.183** |
| e+m+g/mean ∪ lvl | 0.018 | 0.300 | 0.083 | 0.175 | 0.817 | 0.125 |

AutoGen tool_cascade **0.108 → 0.183** at 2.8% false alarms, and the arm exists
at all only because of this change.

**Previous bests preserved**

| corpus / metric | previous best | now | status |
|---|---|---|---|
| langgraph realized drift | 0.800 | 0.800 | held |
| langgraph looping | 1.000 | 1.000 | held |
| langgraph context | 0.658 | 0.650 | −0.008 |
| langgraph tool_cascade | 0.633 | 0.633 | held |
| autogen tool_cascade | 0.167 | **0.183** | improved |
| autogen realized drift | 0.350 | 0.350 | held |
| **long_drift realized drift** | **0.578** | **0.578** | **recovered** |

The long-drift figure needed recovering. Folding the seven reliability dims
into the deviation monitor's dim set cost it 0.578 → 0.554: on a corpus where
tools never fail those dims are inert, and adding 21 inert features shifts the
CUSUM drift constant. The fix is structural rather than a tradeoff — the
reliability channel reaches the detector through its own calibrated branch, and
`DERIVED_DIMS` is left exactly as it was. That restored 0.578 while keeping
every reliability gain, since the gains come from the branch, not the dim set.

## Regressions and unmet targets

- **langgraph context 0.658 → 0.650** and **autogen detection 0.323 → 0.300**
  in the corresponding union arms. Both are inside the replicate-to-replicate
  spread of a 20-draw mean and neither is attributable to a mechanism change;
  the arms involved do not contain the reliability branch.
- **No single arm is best on all three corpora.** `e+m+g ∪ lvl` wins AutoGen,
  `e+m+g ∪ r ∪ lvl` wins LangGraph, `e+m+g ∪ all` wins long drift. This is the
  pre-existing fusion-dilution problem, unchanged by this work and deliberately
  left to a separate scoped change.
- **100% detection of realized failures is not met**: best tool_cascade is
  0.633 (LangGraph, channel alone) and 0.183 (AutoGen).
- Two exploratory arms exceed the 5% budget (`lvl` 0.081 and `all` 0.266 on
  AutoGen) and are not candidates. Every candidate arm sits at 0.012–0.050.

Mechanism: `derail/monitor/normalized_fusion.py`, semantics declared in
`derail/common.py`, contracts in `tests/test_degenerate_channel_scale.py`.
