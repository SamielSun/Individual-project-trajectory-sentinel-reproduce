# Reproduction harness — agent-trajectory-sentinel

A **fork** of [`sunnydubey1111/agent-trajectory-sentinel`](https://github.com/sunnydubey1111/agent-trajectory-sentinel)
(arXiv:2608.02464), branched at upstream commit `1b3e07f`, with a reproduction harness, a controlled
ablation of the paper's fusion rule, a report and a presentation added on top.

The upstream README is preserved as [`UPSTREAM_README.md`](UPSTREAM_README.md).

* **Target:** the paper's headline table *“behavioural monitor — synthetic testbed, five seeds”* —
  `esn_cusum_max` detection / steps saved / episode AUC, over the paper's five seeds.
* **Result:** reproduced. At the master seed every reported metric is identical to the committed
  artifact; the only non-zero difference is the alarm threshold θ, at 5e-11. Over five seeds the
  headline row agrees at the printed precision.
* **Modification:** replace the per-channel **max** fusion with a **mean** (and top-2 mean) on the
  primary monitor only, paired on two seeds the paper never uses.
* **Cost:** 0 LLM calls, \$0, CPU only, offline. 13 arms at ~2.5 min each.

Report: [`report/main.pdf`](report/main.pdf) · Protocol: [`PROTOCOL.md`](PROTOCOL.md) ·
Submission checklist: [`SUBMISSION.md`](SUBMISSION.md) · Presentation: [`slides/`](slides/)

## Reproduce

```powershell
pwsh -File setup.ps1                        # build .venv, install the pinned stack
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m repro.env     # record interpreter, pins, CPU -> out/environment.json
.\.venv\Scripts\python.exe -m repro.run_all # 13 arms -> out/runs/
.\.venv\Scripts\python.exe -m repro.collect # tables + report/numbers.tex
.\.venv\Scripts\python.exe -m repro.figures # report figures
cd report; latexmk -pdf main.tex
```

`python -m repro.run_all --plan` lists the arms without running anything.

## What is compared

`out/runs/<arm>/` holds the `results.json` and `h1_main.csv` that arm produced. The **paper side is
never regenerated in place**: `results/` in this repository is upstream's committed artifact, and every
arm is redirected through the documented `AGENTWATCH_RESULTS_ROOT` hook into `out/runs/_roots/`. So
“ours” and “theirs” stay two independent artifacts, and re-running cannot destroy the thing being
compared against.

| arm | seed(s) | purpose |
|---|---|---|
| `baseline_master` | 20260713 | single-seed reproduction against committed `h1_main.csv` |
| `multiseed_s*` | 7, 101, 202, 303 | completes the five-seed headline table |
| `baseline_s*` | 424242, 31337 | paired control for the modification (seeds the paper never uses) |
| `mean_s*`, `top2_s*` | 424242, 31337 | the modification: max → mean / top-2 mean |
| `budget02_master`, `budget10_master` | 20260713 | sensitivity: same pipeline at a 2 % / 10 % FA budget |

## The modification

One thing changes: how the three per-channel CUSUM streams of the primary monitor are combined.

```python
# upstream, derail/monitor/esn.py:ChannelMaxESNMonitor.score_step
return max(sub.score_step(x_t) for sub in self.subs)

# ours (repro/mods/fusion.py), applied at run time, only for name == "esn_cusum_max"
return float(np.mean([sub.score_step(x_t) for sub in self.subs]))
```

Applied by monkeypatch inside a short-lived subprocess, so the upstream sources stay byte-clean and
`results.json`'s `git_dirty` provenance flag keeps its meaning. Every other monitor keeps `max` and is
an untouched control — `out/tables/t2b_controls_identical.csv` records that they stay bit-identical
under the patch.

## Repository layout

```
derail/ devtools/ verification/ tests/ experimental/   upstream code (unmodified)
results/                                              upstream's committed artifacts (the "paper" side)
repro/config.py            paths, seeds, pinned commit (single source of truth)
repro/run_one.py           run ONE upstream experiment, optionally under a patch
repro/run_all.py           all 13 arms, each in its own subprocess, with provenance
repro/collect.py           join ours vs committed -> out/tables/*.csv + report/numbers.tex
repro/figures.py           report figures
repro/env.py               environment capture
repro/mods/fusion.py       the modification (max -> mean / top-2 mean)
repro/mods/budget.py       sensitivity arm (false-alarm budget)
report/                    LaTeX report; numbers.tex and tab_master_diff.tex are generated
slides/                    generated deck + timecoded speaker script
out/                       generated artifacts (joined tables are committed; raw runs are not)
```

## Notes and gotchas

* **traces/ and assets/ are removed from this fork.** They are ~80 MB of episode JSONL plus README
  media, and the synthetic study this reproduction targets does not read them. Consequence: the 62
  upstream tests that read `traces/*/manifest.json` cannot run here; `PROTOCOL.md` declares that
  instead of hiding it.
* **Python 3.12 is fine.** Upstream's artifacts were produced on 3.14.5; every pinned package ships a
  `cp312` wheel, and the master-seed table reproduces digit-for-digit on 3.12.14 (report §4).
* **Install with `--no-deps` + an explicit transitive list** — see the header of `setup.ps1`.
* **torch is optional upstream.** Without it the GRU/LSTM/TCN rows silently disappear and the table
  still renders; we assert on the row set rather than trusting the file.
* **No secrets.** No API key is used anywhere, and `.env`/`*.key` are gitignored.
