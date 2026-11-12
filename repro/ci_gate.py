"""CI gate: the master-seed reproduction must be exact.

Run after ``python -m repro.run_all --only baseline_master`` and
``python -m repro.collect``. Exits non-zero if any *reported* metric of the
master-seed table differs from the committed artifact.

Why only the master seed: it is the strongest cheap check (column-for-column
equality against upstream's own committed table) and costs one ~2.5 minute arm.
The five-seed table is reproduced the same way but needs five arms; the report
carries it, and `SUBMISSION.md` lists the command.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from repro import config as cfg

# Metrics the paper reports. theta is deliberately excluded: it is a float
# quantile of the healthy validation scores and reproduces to ~1e-11, not to 0.
REPORTED = ("detection_rate", "mean_lead_all", "episode_auc", "healthy_fa_rate",
            "early_alarm_rate")
THETA_TOL = 1e-6


def main() -> int:
    diffs_path = cfg.TABLES_DIR / "t1b_master_seed_column_diffs.csv"
    if not diffs_path.exists():
        print(f"missing {cfg.rel(diffs_path)}: run repro.run_all then repro.collect")
        return 1

    diffs = pd.read_csv(diffs_path).set_index("column")["max_abs_diff"].to_dict()
    missing = [c for c in REPORTED if c not in diffs]
    if missing:
        print(f"the master-seed table is missing reported columns: {missing}")
        return 1

    failures = []
    for col in REPORTED:
        if diffs[col] != 0.0:
            failures.append(f"{col}: max |ours - committed| = {diffs[col]:.3e} (expected 0)")

    theta = diffs.get("theta")
    if theta is not None and theta > THETA_TOL:
        failures.append(f"theta: {theta:.3e} exceeds {THETA_TOL:.0e}")

    results = cfg.RUNS_DIR / "baseline_master" / "results.json"
    if not results.exists():
        failures.append("baseline_master/results.json missing")
    else:
        run = json.loads(results.read_text(encoding="utf-8"))
        pm = run["h1"]["per_monitor"].get("esn_cusum_max", {})
        print(f"esn_cusum_max @ master seed: detection={pm.get('detection_rate')} "
              f"lead_all={pm.get('mean_lead_all')} auc={pm.get('episode_auc')} "
              f"fa={pm.get('healthy_fa_rate')}")

    if failures:
        print("\nREPRODUCTION GATE FAILED")
        for f in failures:
            print("  -", f)
        return 1

    print(f"\nREPRODUCTION GATE PASSED: all {len(REPORTED)} reported metrics of the "
          f"master-seed table match the committed artifact exactly "
          f"(theta differs by {theta:.1e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
