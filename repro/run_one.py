"""Run ONE upstream experiment, optionally under a controlled modification.

Executed as a subprocess by ``repro/run_all.py`` so that every run starts from a
fresh import graph — a monkeypatch can never leak from one arm into the next.

    python -m repro.run_one --seed 424242 --results-root out/runs/_roots/x \
        --mod fusion --fusion mean
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="repro.run_one")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--results-root", required=True,
                    help="where the upstream run writes (AGENTWATCH_RESULTS_ROOT)")
    ap.add_argument("--mod", default="none", choices=("none", "fusion", "budget"))
    ap.add_argument("--fusion", default="mean", choices=("mean", "top2mean"))
    ap.add_argument("--budget", type=float, default=0.05)
    args = ap.parse_args(argv)

    # Import the overlay BEFORE changing directory / sys.path, so `repro` always
    # resolves against the harness repo, never against a same-named upstream module.
    from repro import config as cfg

    upstream = cfg.require_upstream()

    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = (cfg.REPO_ROOT / results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    os.environ["AGENTWATCH_RESULTS_ROOT"] = str(results_root)

    os.chdir(upstream)
    sys.path.insert(0, str(upstream))

    applied = "none"
    if args.mod == "fusion":
        from repro.mods import fusion

        fusion.apply(args.fusion)
        applied = f"fusion:{args.fusion}"
    elif args.mod == "budget":
        from repro.mods import budget

        budget.apply(args.budget)
        applied = f"budget:{args.budget:g}"

    print(f"[repro.run_one] seed={args.seed} mod={applied} "
          f"results_root={results_root}", flush=True)

    from derail.experiments import run_experiment

    if args.mod == "budget":
        print(f"[repro.run_one] FA_BUDGET now {run_experiment.FA_BUDGET:g}", flush=True)

    run_experiment.main(["--seed", str(args.seed)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
