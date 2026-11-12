# Deployment healthy corpus — collection accounting

`traces/real_task_rollback_healthy`, the healthy population the repair
path's own deployment is calibrated on. Collected with
`collect_real_task_rollback_source` against `qwen2.5:7b` and live
arXiv/Wikipedia/Open-Meteo. Admission is the frozen rule: a run is a
healthy negative only if the task's predeclared `success_fn` returns True.

## Predeclared plan

300 slots, 150 per task, fixed before collection and not extended when
attempts were refused:

| leg | seed base | healthy index range | slots |
| --- | --- | --- | --- |
| first | 52026 | 0-39 per task | 80 |
| second | 777 | 40-149 per task | 220 |

Seeds derive from position in the plan, so the second leg uses a fresh
seed base and a shifted index range: existing episodes keep their seeds and
are reused untouched rather than re-collected.

## Outcome

| attempted slots | 300 |
| --- | --- |
| admitted | **297** |
| arxiv_paper_search | 147 |
| multi_city_weather | 150 |
| refused, never admitted | 2 |
| admitted then excluded | 1 |
| episode length T | min 4, median 6, max 12 |

Every admitted episode carries `failure_class = null` and `success = true`.

## Refused slots

| slot | attempts | reason |
| --- | --- | --- |
| `rollback-arxiv_paper_search-healthy-60` | 2 | `too short: T=2 < 4`, then `T=3 < 4` |
| `rollback-arxiv_paper_search-healthy-101` | 2 | `too short: T=2 < 4` both times |

Neither slot was replaced and no substitute episode was collected.

## One slot excluded after a passing retry

`rollback-arxiv_paper_search-healthy-96` failed the independent task
verifier on its first attempt, in a healthy environment -- its cassette
records three successful tool calls, so nothing external prevented the run
from succeeding. The collector re-attempts any slot missing from the
manifest, so the slot ran a second time and passed.

It is **not** admitted. A slot whose valid attempt failed the verifier does
not become a healthy negative because a later roll of the same seed passed;
admitting it would select the population on outcome. Both attempts and the
original rejection reason are kept in `rejected.json`, and the trace file
and cassettes stay on disk. Only the manifest entry is removed, which is
what keeps it out of every fold.

## A collection interruption, diagnosed

The second leg stopped partway with eight consecutive refusals, seven of
them `too short: T=2 < 4`. Those episodes recorded no tool calls at all.
Errors are deliberately not cached, so an episode whose every tool call
failed leaves an empty cassette -- which is what those episodes have. The
cause was the external service, not the agent and not the code: all three
tools were re-verified live before collection resumed, and seven of those
eight slots were admitted on re-attempt.

## Folds

Disjoint, at the committed floors from `results/calibration_sample_size.md`:

| fit | 140 |
| --- | --- |
| threshold calibration | 80 |
| untouched development | 77 |

The untouched fold is development evidence. It is **not** the final
held-out live E2E test population and must never be reported as one.
