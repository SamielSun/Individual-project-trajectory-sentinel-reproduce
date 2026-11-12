# Submission checklist

Current state: the experiments are run and committed, the repository is pushed to
`github.com/SamielSun/Individual-project-trajectory-sentinel-reproduce`, and the recording is in the
tree. The two things below are the only ones left.

## If you need to push again

```powershell
cd "D:\CUHK Fintech\FTEC5660HW\sentinel-fork"
git status                    # expect a clean tree
git log --oneline             # 19 commits, newest first
git push
```

The repository is about 27 MB, including the 12 MB recording.

Before pushing anything new, check that no working credential slipped in:

```powershell
git log --all -p | Select-String -Pattern "sk-|api[_-]?key" -CaseSensitive:$false
```

This does return matches, and they are upstream's test fixtures (for example
`sk-abcdefghijklmnopqrstuvwxyz012345` in `tests/test_hf_dataset.py`). What matters is that nothing new
appears in your own commits.

## If you re-record or upload the talk elsewhere

The report title block currently links to `slides/presentation.mp4` (8:48). To point it at a YouTube or
OneDrive URL instead, edit that line in `report/main.tex`, then:

```powershell
cd report; latexmk -pdf main.tex; cd ..
```

Nothing else changes.

## Before submitting

* `report/main.pdf` — check the name, student ID and both links on page 1
* `slides/presentation.mp4` — 8:48, inside the 15-minute cap
* repository link

To confirm the report's numbers still match the artifacts:

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m repro.collect
git diff --stat report/numbers.tex out/tables     # empty if nothing is stale
```

## Where each brief requirement is answered

| Brief requirement | Where |
|---|---|
| rule 1: public repo with a runnable pipeline | this fork; `setup.ps1` + `python -m repro.run_all` |
| rule 2: one small task, one table row set | `PROTOCOL.md` "Target claim", report Section 1 "Scope" |
| rule 3: not re-implementing from scratch | upstream modules run unmodified; the patch is one aggregation function |
| rule 4: model swap allowed | not needed here; the reproduced table is the monitor's, and Section 2 of the report declares the scope |
| rule 5: one benchmark, 3–5 trials, mean and variance | five seeds, mean ± sample std (`ddof=1`), `out/tables/t1_*` |
| rule 6: one isolated, measurable change | `repro/mods/fusion.py`, report Section 5, `out/tables/t2_*` |
| rule 7: report, GitHub link, presentation | `report/main.pdf`, this repository, `slides/presentation.mp4` |

## Numbers you will be asked about

| file | what it shows |
|---|---|
| `out/tables/t1_multiseed_mine_vs_paper.csv` | five-seed table, ours against the committed one |
| `out/tables/t1b_master_seed_column_diffs.csv` | master-seed column-by-column diff (zero except the threshold, 5e-11) |
| `out/tables/t2_fusion_ablation.csv` | max vs mean vs top-2 mean, paired on two unseen seeds |
| `out/tables/t2b_controls_identical.csv` | 36 control comparisons, all unchanged |
| `out/tables/t3_fa_budget.csv` | false-alarm-budget sensitivity (2 % / 5 % / 10 %) |
