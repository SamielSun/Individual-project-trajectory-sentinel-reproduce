NOTICE — reproduction harness
============================

This repository is a fork of

    AgentTrajectorySentinel — "Real-Time Detection and Repair of LLM Agent Failures"
    https://github.com/sunnydubey1111/agent-trajectory-sentinel
    Copyright the upstream authors. Licensed under the Apache License, Version 2.0.
    Fork point: commit 1b3e07f (see `git log --oneline | tail`).

Everything the upstream project shipped keeps its own licence and its own NOTICE
(`NOTICE`); this file covers only what the fork adds.

Added by this fork (Apache-2.0, so it can be combined with the upstream tree)
------------------------------------------------------------------------------
`repro/`, `report/`, `slides/`, `setup.ps1`, `README.md`, `PROTOCOL.md`,
`SUBMISSION.md`, `NOTICE.reproduction.md` and the files in `out/tables/`.

Removed by this fork
--------------------
`traces/` (79 MB of episode JSONL) and `assets/` (README media). Neither is read
by the synthetic study this reproduction targets. The 62 upstream tests that
read `traces/*/manifest.json` therefore cannot pass in this fork; that is
declared in `PROTOCOL.md` rather than hidden.

Modification of upstream behaviour
----------------------------------
Section 4(b) and 4(d) of the Apache-2.0 licence require modified files to carry
prominent notices of the change. No upstream file is modified on disk. The one
behavioural change is applied by monkeypatch inside a short-lived subprocess, so
it is marked here:

    Modified class: derail.monitor.esn.ChannelMaxESNMonitor
    Changed:        score_step and score_episode_components aggregate the three
                    per-channel streams with mean (or top-2 mean) instead of max,
                    and only for the monitor named "esn_cusum_max".
    When:           only in processes started with `--mod fusion`
                    (baseline arms are unmodified upstream behaviour, byte for byte).
    Evidence:       out/tables/t2_fusion_ablation.csv and
                    out/tables/t2b_controls_identical.csv.

Scientific provenance
---------------------
The fork is based on upstream commit 1b3e07f, whose committed `results/` tree is
what this reproduction compares against. The arXiv v1 preprint describes an
earlier tree, commit 00c0673 (2,823 episodes / 25 corpora, versus 4,022 / 35 at
1b3e07f). Any table quoted against "the paper" must state which one it used;
`out/environment.json` records the commit for every run.
