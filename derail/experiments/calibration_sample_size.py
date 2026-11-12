"""How many healthy episodes does a per-deployment calibration actually need?

`deployment_calibration` shipped two numbers -- refuse below 30, warn below
80 -- attributed to a native-harness learning curve that was never committed.
This is that curve, run and recorded, so the floors are evidence rather than
recollection.

Three sample sizes are varied and reported separately, because they answer
different questions and the 60/20/20 split hides that they are not one number:

  fit N          healthy episodes the standardizer and reservoir see;
  calibration N  healthy episodes the threshold is picked from;
  test N         untouched healthy episodes the realized false-alarm rate is
                 measured on, which never touches a fit or a threshold.

The test set here is drawn ONCE, before anything else, and no fitted or
thresholded episode is ever drawn from it. Every arm is scored on that same
population, so the columns are comparable.

Also reports what precision a false-alarm claim needs, which is a property of
the test N alone: with k false alarms in n untouched healthy episodes, the
Clopper-Pearson upper bound is what an operational "<= 5%" statement can
honestly say.

Writes:
  results/tables/calibration_sample_size.csv
  results/tables/calibration_fa_precision.csv
  results/calibration_sample_size.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from derail.common import rng_for
from derail.evaluation.metrics import first_alarm
from derail.monitor.deployment_calibration import calibrate_from_folds
from derail.telemetry.adapter import episode_from_trace

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_ROOT = REPO_ROOT / "traces"
OUT_CSV = REPO_ROOT / "results" / "tables" / "calibration_sample_size.csv"
OUT_PRECISION = REPO_ROOT / "results" / "tables" / "calibration_fa_precision.csv"
OUT_REPORT = REPO_ROOT / "results" / "calibration_sample_size.md"

#: The native-harness healthy population the shipped floors claimed to come
#: from -- the freeze's own fit corpora, so the curve describes the same
#: population the deployment procedure was designed against.
POOL_CORPORA = ("real_research7b", "real_ollama7b", "real_research7b_long",
                "real_research7b_long_ext", "real_research7b_long_drift")

CHANNELS = ("e", "m")
FA_BUDGET = 0.05
N_TEST = 100
FIT_SIZES = (20, 40, 80, 140)
CAL_SIZES = (10, 20, 40, 80)
REPLICATES = 10
SPLIT_SEED = 90124


def load_pool() -> list:
    episodes = []
    for corpus in POOL_CORPORA:
        corpus_dir = TRACES_ROOT / corpus
        manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
        for entry in sorted((e for e in manifest if e["failure_class"] is None),
                            key=lambda e: e["episode_id"]):
            steps = [json.loads(line) for line in
                     (corpus_dir / entry["file"]).read_text("utf-8").splitlines()
                     if line.strip()]
            episodes.append(episode_from_trace(
                steps, f"{corpus}/{entry['episode_id']}",
                use_sentence_transformers=False, extended=True))
    return episodes


def fa_precision_table() -> pd.DataFrame:
    """What an operational false-alarm claim can say at each test size.

    Independent of any monitor: it is the arithmetic of a binomial interval,
    and it is what decides how large the final untouched healthy population
    has to be BEFORE any of it is collected.
    """
    rows = []
    for n in (12, 24, 40, 59, 80, 100, 120, 160, 200, 300):
        for observed in (0.0, 0.02, 0.05):
            k = int(round(observed * n))
            ci = sps.binomtest(k, n).proportion_ci(confidence_level=0.95,
                                                   method="exact")
            rows.append({"n_untouched_healthy": n, "observed_fa": k / n,
                         "k": k, "ci_lo": float(ci.low),
                         "ci_hi": float(ci.high),
                         "supports_5pct_claim": bool(ci.high <= 0.05)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell dispersion and realized false alarms.

    Dispersion is reported as p90/p10 rather than max/min: with ten
    replicates a single runaway threshold makes the extreme ratio a statement
    about that one draw. The maximum is not hidden -- `theta p90` and the
    worst realized rate are both columns -- but the headline number should
    describe the cell.
    """
    ok = df[df["theta"].notna()]
    return ok.groupby(["fit_n", "cal_n"]).agg(
        n=("theta", "size"),
        theta_p10=("theta", lambda s: float(np.quantile(s, 0.10))),
        theta_median=("theta", "median"),
        theta_p90=("theta", lambda s: float(np.quantile(s, 0.90))),
        theta_p90_over_p10=("theta", lambda s: float(
            np.quantile(s, 0.90) / max(np.quantile(s, 0.10), 1e-12))),
        fa_mean=("test_fa_rate", "mean"),
        fa_max=("test_fa_rate", "max"),
        within_budget=("test_fa_rate", lambda s: float((s <= FA_BUDGET).mean())),
    ).reset_index()


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from the committed CSV without "
                         "refitting anything")
    args = ap.parse_args(argv)
    if args.report_only:
        write_report(pd.read_csv(OUT_CSV), pool_size=None, test_size=None)
        return
    pool = load_pool()
    order = rng_for(SPLIT_SEED, "calibration-sample-size").permutation(len(pool))
    test = [pool[i] for i in order[:N_TEST]]
    dev = [pool[i] for i in order[N_TEST:]]
    print(f"[cal-size] pool={len(pool)} untouched test={len(test)} "
          f"development={len(dev)}")

    rows = []
    for replicate in range(REPLICATES):
        perm = rng_for(SPLIT_SEED, "replicate", replicate).permutation(len(dev))
        shuffled = [dev[i] for i in perm]
        for fit_n in FIT_SIZES:
            fit = shuffled[:fit_n]
            rest = shuffled[fit_n:]
            for cal_n in CAL_SIZES:
                if cal_n > len(rest):
                    continue
                cal = rest[:cal_n]
                try:
                    result = calibrate_from_folds(
                        fit, cal, channels=CHANNELS, fa_budget=FA_BUDGET,
                        seed=replicate)
                except ValueError as exc:
                    rows.append({"replicate": replicate, "fit_n": fit_n,
                                 "cal_n": cal_n, "refused": str(exc)[:100]})
                    continue
                alarms = 0
                for ep in test:
                    result.monitor.start_episode()
                    stream = np.array([result.monitor.score_step(ep.X[t])
                                       for t in range(ep.T)])
                    alarms += first_alarm(stream, result.theta) is not None
                rows.append({
                    "replicate": replicate, "fit_n": fit_n, "cal_n": cal_n,
                    "theta": result.theta,
                    "order_statistic_floor": 1.0 / (cal_n + 1),
                    "n_test": len(test), "n_fa": alarms,
                    "test_fa_rate": alarms / len(test),
                })
            print(f"[cal-size] replicate {replicate} fit_n={fit_n} done")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    write_report(df, pool_size=len(pool), test_size=len(test))


def write_report(df: pd.DataFrame, *, pool_size: int | None,
                 test_size: int | None) -> None:
    precision = fa_precision_table()
    precision.to_csv(OUT_PRECISION, index=False)
    grouped = summarize(df)
    pool_text = ("" if pool_size is None else
                 f"Pool: {pool_size} healthy episodes from "
                 f"{', '.join(POOL_CORPORA)}. ")
    test_text = (f"{test_size}" if test_size is not None
                 else f"{int(df['n_test'].iloc[0])}")

    lines = ["# How many healthy episodes a calibration needs",
             "",
             "**SUPPORTING / diagnostic evidence.** Replaces the uncommitted "
             "curve behind `_MIN_HEALTHY = 30` and `_RECOMMENDED_HEALTHY = "
             "80`. No published number depends on this; it exists to choose "
             "the sample sizes the final evaluation will be built on, before "
             "that evaluation is collected.",
             "",
             f"{pool_text}A single untouched test set of {test_text} was "
             f"drawn first and is never fitted or thresholded on; every row "
             f"below is scored against it. Channels {CHANNELS}, budget "
             f"{FA_BUDGET:.0%}, {REPLICATES} replicates per cell with "
             f"independent fit and calibration draws.",
             "",
             "Dispersion is `theta p90 / theta p10` across replicates: how "
             "far the same procedure moves on nothing but which healthy "
             "episodes it happened to see.",
             "",
             "| fit N | calibration N | replicates | theta p10 | theta "
             "median | theta p90 | p90/p10 | mean test FA | worst test FA | "
             "replicates within budget |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in grouped.iterrows():
        lines.append(
            f"| {int(r['fit_n'])} | {int(r['cal_n'])} | {int(r['n'])} | "
            f"{r['theta_p10']:.1f} | {r['theta_median']:.1f} | "
            f"{r['theta_p90']:.4g} | {r['theta_p90_over_p10']:.1f}x | "
            f"{r['fa_mean']:.3f} | {r['fa_max']:.3f} | "
            f"{r['within_budget']:.0%} |")

    worst = df.loc[df["theta"].idxmax()]
    lines += ["",
              "A large `theta p90` beside a low false-alarm rate is not a "
              "good row. A threshold that has run away alarms on nothing and "
              "earns a false-alarm rate of zero by refusing to detect "
              "anything either, so the two columns are only meaningful read "
              f"together: the most extreme draw here is fit "
              f"{int(worst['fit_n'])} / calibration {int(worst['cal_n'])}, "
              f"which picked theta = {worst['theta']:.3g} and then false "
              f"alarmed on {worst['test_fa_rate']:.0%} of the untouched "
              f"healthy set.",
              "",
              "## What a false-alarm claim needs from the test set", "",
              "Independent of the monitor: with `k` false alarms in `n` "
              "untouched healthy episodes, this is the exact 95% interval. It "
              "is what decides the final healthy test size before collection.",
              "",
              "| untouched healthy N | observed FA | 95% CI | supports a "
              "<=5% claim |", "|---|---|---|---|"]
    for _, r in precision.iterrows():
        lines.append(
            f"| {int(r['n_untouched_healthy'])} | {r['observed_fa']:.3f} "
            f"({int(r['k'])}) | {r['ci_lo']:.3f}-{r['ci_hi']:.3f} | "
            f"{'yes' if r['supports_5pct_claim'] else 'no'} |")
    lines.append("")
    OUT_REPORT.write_text("\n".join(lines), "utf-8")
    print(grouped.to_string(index=False))
    for path in (OUT_CSV, OUT_PRECISION, OUT_REPORT):
        print(f"[cal-size] wrote {path}")


if __name__ == "__main__":
    main()
