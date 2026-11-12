"""Run every arm of the reproduction and file the artifacts under out/runs/.

Design choices that matter for the report
------------------------------------------
* Every run is a separate subprocess (``repro.run_one``) so a monkeypatch cannot
  leak between arms and every run reports its own provenance.
* Every run writes through ``AGENTWATCH_RESULTS_ROOT`` into ``out/runs/_roots``,
  so the upstream ``results/`` tree — the paper's own committed artifact — is
  never overwritten. That is what makes "my numbers vs the paper's numbers" a
  comparison of two independent artifacts rather than of a file with itself.
* The five-seed summary is recomputed here with the authors' aggregation rule
  (mean and sample std, ddof=1) instead of calling ``run_multiseed``, because
  that module hard-codes ``results/`` as its base and would ignore the redirect.

Usage
-----
    python -m repro.run_all --sets all
    python -m repro.run_all --sets baseline multiseed
    python -m repro.run_all --plan            # print the arm list, run nothing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from repro import config as cfg

ARMS_TIMEOUT_S = 1800

# arm name -> (seed, mod, mod-args)
def build_arms(sets: set[str]) -> list[dict]:
    arms: list[dict] = []

    if "baseline" in sets:
        arms.append({"name": "baseline_master", "seed": cfg.MASTER_SEED,
                     "mod": "none", "mod_args": {}})
        for seed in cfg.PAIRED_SEEDS:
            arms.append({"name": f"baseline_s{seed}", "seed": seed,
                         "mod": "none", "mod_args": {}})

    if "multiseed" in sets:
        for seed in cfg.MULTISEED_SEEDS:
            if seed == cfg.MASTER_SEED:
                continue  # the master-seed arm is produced by "baseline"
            arms.append({"name": f"multiseed_s{seed}", "seed": seed,
                         "mod": "none", "mod_args": {}})

    if "modification" in sets:
        for seed in cfg.PAIRED_SEEDS:
            arms.append({"name": f"mean_s{seed}", "seed": seed,
                         "mod": "fusion", "mod_args": {"fusion": "mean"}})
            arms.append({"name": f"top2_s{seed}", "seed": seed,
                         "mod": "fusion", "mod_args": {"fusion": "top2mean"}})

    if "budget" in sets:
        for b in (0.02, 0.10):
            arms.append({"name": f"budget{int(b * 100):02d}_master",
                         "seed": cfg.MASTER_SEED, "mod": "budget",
                         "mod_args": {"budget": b}})

    return arms


def run_arm(arm: dict) -> dict:
    name, seed, mod = arm["name"], arm["seed"], arm["mod"]
    root = cfg.RUNS_DIR / "_roots" / name
    outdir = cfg.run_dir(name)
    if root.exists():
        shutil.rmtree(root)

    cmd = [cfg.upstream_python(), "-m", "repro.run_one",
           "--seed", str(seed), "--results-root", str(root), "--mod", mod]
    if arm["mod_args"].get("fusion"):
        cmd += ["--fusion", arm["mod_args"]["fusion"]]
    if arm["mod_args"].get("budget") is not None:
        cmd += ["--budget", str(arm["mod_args"]["budget"])]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(cfg.REPO_ROOT)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[run] {name}: seed={seed} mod={mod} {arm['mod_args'] or ''}", flush=True)
    t0 = time.time()
    # The child's output goes straight to a file, never through a pipe: the
    # upstream study keeps long-lived worker threads alive, and a captured pipe
    # the parent stops draining deadlocks the whole arm (observed once, ~30 min
    # of wall clock for a 148 s run). A file handle cannot deadlock.
    log_path = outdir / "stdout.log"
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        try:
            proc = subprocess.run(cmd, cwd=str(cfg.REPO_ROOT), env=env,
                                  stdout=logf, stderr=subprocess.STDOUT,
                                  timeout=ARMS_TIMEOUT_S)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
            logf.write(f"\n[repro] arm exceeded {ARMS_TIMEOUT_S}s and was killed\n")
    dt = time.time() - t0

    # Where the upstream run put its files: the master seed writes at the root,
    # any other seed under seed<N>/ (see run_experiment._set_results_dirs).
    run_root = root if seed == cfg.MASTER_SEED else root / f"seed{seed}"
    copied = {}
    for fname in ("results.json", "tables/h1_main.csv", "tables/h2_channels.csv",
                  "tables/delay_fa_curves.csv"):
        src = run_root / fname
        if src.exists():
            dst = outdir / Path(fname).name
            shutil.copy2(src, dst)
            copied[Path(fname).name] = src.stat().st_size

    meta = {
        "name": name,
        "seed": seed,
        "mod": mod,
        "mod_args": arm["mod_args"],
        "returncode": rc,
        "duration_s": round(dt, 1),
        "artifacts": copied,
        "results_root": cfg.rel(run_root),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[run] {name}: rc={rc} {dt:.1f}s files={list(copied)}", flush=True)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="repro.run_all")
    ap.add_argument("--sets", nargs="+",
                    default=["baseline", "multiseed", "modification", "budget"],
                    choices=("baseline", "multiseed", "modification", "budget", "all"))
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these arm names (debugging)")
    args = ap.parse_args(argv)

    cfg.require_upstream()
    sets = set(args.sets)
    if "all" in sets:
        sets = {"baseline", "multiseed", "modification", "budget"}

    arms = build_arms(sets)
    if args.only:
        arms = [a for a in arms if a["name"] in set(args.only)]

    if args.plan:
        for a in arms:
            print(f"{a['name']:<20s} seed={a['seed']:<10d} mod={a['mod']:<8s} {a['mod_args']}")
        print(f"{len(arms)} arms")
        return 0

    cfg.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for arm in arms:
        manifest.append(run_arm(arm))
        (cfg.RUNS_DIR / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

    bad = [m for m in manifest if m["returncode"] != 0]
    print(f"\n{len(manifest) - len(bad)}/{len(manifest)} arms OK, {len(bad)} failed")
    for m in bad:
        print(f"  FAILED {m['name']} (see out/runs/{m['name']}/stdout.log)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
