# Reproduction protocol

Fixed before the first arm was run, so the numbers could not choose the design.

## Target claim

From the paper's README table *“Behavioural monitor — synthetic testbed, five seeds”*
(5 % healthy-validation false-alarm budget):

| monitor | detection | steps saved | AUC |
|---|---|---|---|
| `esn_cusum_max` | 0.71 ± 0.07 | 4.6 ± 1.0 | 0.872 ± 0.015 |

and one mechanism claim the table is used to support:

> “the per-channel max-fusion wrapper, not the reservoir, carries most of the margin”

## Definitions (taken from the upstream evaluation module, not re-derived)

* **detection rate** — fraction of injected-failure test episodes with a first alarm strictly after
  onset τ and before the episode ends. Causal: only what was observable by the alarm step.
* **steps saved** (`mean_lead_all`) — expected budget saved per failure episode, a miss counting 0
  rather than being dropped (survivorship-free).
* **episode AUC** — offline ranking statistic (per-episode max score vs the healthy population). Never
  reported as detection; rises with episode length.
* **realised FA** — fraction of healthy *test* episodes alarming at a threshold picked on healthy
  *validation* for a 5 % budget. Asked-for budget and realised rate are reported separately.

Splits and discipline are the upstream ones: monitors fit on healthy train only, thresholds from
healthy val only, every reported number from test, five seeds `{20260713, 7, 101, 202, 303}`, aggregate
= mean ± sample std (`ddof=1`).

## Reproduction criteria

1. **Exact (master seed).** Our `h1_main.csv` must match the committed file column-for-column,
   max |diff| = 0.
2. **Headline (five seeds).** Our recomputed mean/std must match the committed
   `multiseed_summary.csv` to the printed precision (3 decimals for means).
3. Anything that fails (1) or (2) is reported as a finding, not smoothed over.

## Modification (fixed before running)

Single change, applied at run time, only to `esn_cusum_max`:
`max(score_e, score_u, score_m)` → `mean(...)` (plus `top2mean` as a graded control).
Everything else fixed: same sub-monitors, reservoir weights, seeds, CUSUM, standardiser, splits,
threshold rule, and the same 5 % budget.

* Paired design: both arms on the same two disposable seeds (`424242`, `31337`) never used by the
  published study.
* Falsification: if the margin barely moves when max is replaced, the paper's attribution of the gain
  to the fusion wrapper is wrong; if it collapses, the attribution is supported.
* Isolation check: all other monitor rows must be bit-identical under the patch
  (`out/tables/t2b_controls_identical.csv`).

## Out of scope (declared, not silently skipped)

Live repair arm (needs a served local model), AFTraj-2K / ATBench (external corpora), Gemini judge
calibration (paid API), wall-clock latency benchmark (machine property; the paper itself calls it the
one non-bit-reproducible number), GRU/LSTM/TCN rows (torch not installed — the row set shrinks
silently, so it is asserted against `MULTI_MONITORS` instead of assumed).

## Environment deviations (recorded before running)

| item | paper | this run |
|---|---|---|
| OS | Windows 11 | Windows 11 |
| interpreter | CPython 3.14.5 | CPython 3.12.14 |
| numpy / scipy / pandas / sklearn / matplotlib | 2.4.6 / 1.17.1 / 3.0.3 / 1.9.0 / 3.11.0 | identical |
| torch | 2.12.0+cpu | absent (optional import) |
| GPU | none | none |
| network / API | none for the synthetic study | none at run time |
