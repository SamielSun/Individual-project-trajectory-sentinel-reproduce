"""Is `_WASHOUT = 3` responsible for the short-episode misses?

Diagnosis only, on development data. The washout question was closed once
before on `real_research7b` as a negative for episode AUC and lead time; this
re-opens only the specific case the framework corpora exposed -- injected
episodes with tau = 2 and T = 4-5, where a three-step washout leaves one or
two scored steps for the CUSUM to work with.

Sweeps the washout over 0..3 and, independently, the CUSUM slack k, refitting
the whole per-deployment calibration at each setting so nothing is compared
against a monitor fit under a different rule. Reports what each setting costs
in healthy false alarms as well as what it buys in detection, because a
washout of 0 also lets the reservoir's untrained transient into the healthy
null.

Writes:
  results/tables/short_horizon_sweep.csv
  results/short_horizon_diagnosis.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from derail.evaluation.metrics import first_alarm
from derail.experiments.diagnose_framework_misses import (FA_BUDGET,
                                                          FRAMEWORKS,
                                                          SHIPPED_CHANNELS,
                                                          channel_trace,
                                                          load_corpus)
from derail.monitor import esn as esn_module
from derail.monitor.deployment_calibration import calibrate

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "results" / "tables" / "short_horizon_sweep.csv"
OUT_PER_EPISODE = (REPO_ROOT / "results" / "tables" /
                   "short_horizon_per_episode.csv")
OUT_REPORT = REPO_ROOT / "results" / "short_horizon_diagnosis.md"

WASHOUTS = (0, 1, 2, 3)
#: An episode is short-horizon when onset leaves it at most this many steps
#: to be detected in. Stated as a property of the trajectory, so it selects
#: the same episodes in any corpus and never names one.
SHORT_HORIZON_POST_ONSET = 3
#: The reference value the CUSUM subtracts from every increment. At the
#: shipped 0.5 a single scored step whose evidence is worth less than 0.5
#: leaves the statistic at exactly zero.
SLACKS = (0.0, 0.1, 0.25, 0.5)


def _score(monitor, episode) -> np.ndarray:
    monitor.start_episode()
    return np.array([monitor.score_step(episode.X[t])
                     for t in range(episode.T)])


def _evaluate(healthy: list, injected: list, *, cusum_k: float) -> dict:
    cal = calibrate(healthy, channels=SHIPPED_CHANNELS, fa_budget=FA_BUDGET,
                    seed=0, cusum_k=cusum_k)
    test = [ep for ep in healthy if ep.episode_id in cal.test_ids]
    fa = sum(1 for ep in test
             if first_alarm(_score(cal.monitor, ep), cal.theta) is not None)
    per_class: dict[str, list[int]] = {}
    detected = 0
    for ep in injected:
        alarm = first_alarm(_score(cal.monitor, ep), cal.theta)
        hit = int(alarm is not None and alarm >= ep.tau)
        detected += hit
        per_class.setdefault(ep.failure_class, []).append(hit)
    silent = sum(1 for ep in injected
                 if _score(cal.monitor, ep).max() == 0.0)
    return {"theta": cal.theta, "n_test": len(test), "n_fa": fa,
            "fa_rate": fa / len(test), "n_injected": len(injected),
            "detection_rate": detected / len(injected),
            "n_all_zero_injected": silent,
            "per_class": {c: (sum(v), len(v)) for c, v in per_class.items()}}


def _per_episode(healthy: list, injected: list, framework: str,
                 corpus: str, washout: int) -> list[dict]:
    """One row per short-horizon injected episode at this washout.

    Aggregate detection can stay flat while individual episodes move in
    opposite directions, so the washout question is settled per episode:
    how many causal post-onset steps existed, what each channel scored on
    them, what the decision statistic reached, and whether it alarmed.
    """
    cal = calibrate(healthy, channels=SHIPPED_CHANNELS, fa_budget=FA_BUDGET,
                    seed=0)
    rows = []
    for ep in injected:
        if ep.T - ep.tau > SHORT_HORIZON_POST_ONSET:
            continue
        traces = {sub.channels[0]: channel_trace(sub, ep)
                  for sub in cal.monitor.subs}
        fused = np.max([tr["cusum"] for tr in traces.values()], axis=0)
        alarm = first_alarm(fused, cal.theta)
        first_scored = washout if ep.T > washout else None
        row = {"dataset": corpus, "framework": framework,
               "episode_id": ep.episode_id, "failure_class": ep.failure_class,
               "T": ep.T, "tau": ep.tau, "washout": washout,
               "first_scored_step": first_scored,
               "causal_post_onset_steps": max(0, ep.T - max(ep.tau, washout)),
               "theta": cal.theta,
               "decision_statistic_max_cusum": float(fused.max()),
               "margin_to_theta": float(fused.max() - cal.theta),
               "alarm_step": float("nan") if alarm is None else float(alarm),
               "detected": bool(alarm is not None and alarm >= ep.tau)}
        for name, tr in traces.items():
            row[f"{name}_cusum_by_step"] = json.dumps(
                [round(float(v), 3) for v in tr["cusum"]])
            row[f"{name}_raw_increment_by_step"] = json.dumps(
                [round(float(v), 3) for v in tr["raw"]])
        row["fused_cusum_by_step"] = json.dumps(
            [round(float(v), 3) for v in fused])
        rows.append(row)
    return rows


def main() -> None:
    original_washout = esn_module._WASHOUT
    rows, episode_rows = [], []
    try:
        for corpus, framework in FRAMEWORKS.items():
            episodes, _ = load_corpus(corpus)
            healthy = [ep for ep in episodes if ep.is_healthy]
            injected = [ep for ep in episodes if not ep.is_healthy]
            for washout in WASHOUTS:
                if washout > 0:
                    esn_module._WASHOUT = washout
                    episode_rows += _per_episode(healthy, injected, framework,
                                                 corpus, washout)
                for slack in SLACKS:
                    esn_module._WASHOUT = washout
                    try:
                        m = _evaluate(healthy, injected, cusum_k=slack)
                    except ValueError as exc:       # degenerate threshold
                        rows.append({"framework": framework,
                                     "washout": washout, "cusum_k": slack,
                                     "error": str(exc)[:120]})
                        continue
                    row = {"framework": framework, "washout": washout,
                           "cusum_k": slack, "theta": m["theta"],
                           "n_healthy_test": m["n_test"], "n_fa": m["n_fa"],
                           "healthy_fa_rate": m["fa_rate"],
                           "n_injected": m["n_injected"],
                           "detection_rate": m["detection_rate"],
                           "n_all_zero_injected": m["n_all_zero_injected"]}
                    for cls, (hit, n) in sorted(m["per_class"].items()):
                        row[f"det_{cls}"] = hit / n
                    rows.append(row)
                    print(f"[short-horizon] {framework} washout={washout} "
                          f"k={slack}: det={m['detection_rate']:.3f} "
                          f"fa={m['fa_rate']:.3f} theta={m['theta']:.2f}")
    finally:
        esn_module._WASHOUT = original_washout

    df = pd.DataFrame(rows)
    per_episode = pd.DataFrame(episode_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    per_episode.to_csv(OUT_PER_EPISODE, index=False)

    lines = ["# Short-horizon diagnosis: washout and CUSUM slack",
             "",
             "**SUPPORTING / diagnostic evidence, on development data.** "
             "Every row refits the whole per-deployment calibration at that "
             "setting, on the same healthy fold, at the same 5% budget. "
             "Shipped setting is washout 3, k 0.5. Nothing here selects a "
             "value: a setting chosen because it maximises a column of this "
             "table would be tuned on the corpus it is measured on.",
             ""]
    for framework in sorted(df["framework"].unique()):
        sub = df[df["framework"] == framework]
        lines += [f"## {framework}", "",
                  "| washout | k | theta | detection | healthy FA | "
                  "injected episodes scoring 0 |", "|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            if "error" in r and isinstance(r.get("error"), str):
                lines.append(f"| {int(r['washout'])} | {r['cusum_k']} | "
                             f"refused | -- | -- | {r['error']} |")
                continue
            lines.append(
                f"| {int(r['washout'])} | {r['cusum_k']} | {r['theta']:.2f} | "
                f"{r['detection_rate']:.3f} | {r['healthy_fa_rate']:.3f} | "
                f"{int(r['n_all_zero_injected'])} |")
        lines.append("")
    lines += ["## Per-episode: every short-horizon injected episode", "",
              f"Selected by trajectory shape alone -- at most "
              f"{SHORT_HORIZON_POST_ONSET} steps between onset and the end of "
              f"the episode -- so the same rule picks the same episodes in "
              f"any corpus. Aggregate detection can hold still while "
              f"individual episodes move in opposite directions, so the "
              f"washout question is settled here rather than in the tables "
              f"above.",
              "",
              "| framework | episode | class | T | tau | washout | causal "
              "post-onset steps | first scored | max CUSUM | theta | alarm | "
              "detected |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in per_episode.sort_values(
            ["framework", "episode_id", "washout"]).iterrows():
        alarm = "-" if pd.isna(r["alarm_step"]) else int(r["alarm_step"])
        first = ("-" if pd.isna(r["first_scored_step"])
                 else int(r["first_scored_step"]))
        lines.append(
            f"| {r['framework']} | {r['episode_id']} | {r['failure_class']} | "
            f"{r['T']} | {r['tau']} | {r['washout']} | "
            f"{r['causal_post_onset_steps']} | {first} | "
            f"{r['decision_statistic_max_cusum']:.2f} | {r['theta']:.2f} | "
            f"{alarm} | {'yes' if r['detected'] else 'no'} |")
    # Verdict, computed rather than asserted: for each episode the shipped
    # washout misses, does ANY washout detect it, and how close does the
    # statistic get to that washout's own threshold? Lowering the washout
    # raises the statistic and the threshold together, so the ratio is the
    # quantity that decides the question, not the raw score.
    shipped = per_episode[per_episode["washout"] == esn_module._WASHOUT]
    missed = set(shipped[~shipped["detected"]]["episode_id"])
    lines += ["", "### Verdict per episode the shipped washout misses", "",
              "`margin ratio` is the decision statistic over that washout's "
              "own threshold; 1.0 is an alarm. It is the ratio and not the "
              "raw score that decides, because a shorter washout raises both.",
              "",
              "| episode | T | tau | ratio @w1 | ratio @w2 | ratio @w3 | "
              "any washout detects |", "|---|---|---|---|---|---|---|"]
    for eid in sorted(missed):
        rows_e = per_episode[per_episode["episode_id"] == eid]
        ratios = {int(r["washout"]):
                  r["decision_statistic_max_cusum"] / r["theta"]
                  for _, r in rows_e.iterrows()}
        first = rows_e.iloc[0]
        lines.append(
            f"| {eid} | {first['T']} | {first['tau']} | " +
            " | ".join(f"{ratios.get(w, float('nan')):.2f}"
                       for w in (1, 2, 3)) +
            f" | {'yes' if bool(rows_e['detected'].any()) else 'no'} |")
    gained = sorted(set(per_episode[per_episode["detected"]]["episode_id"])
                    & missed)
    lost = sorted(set(shipped[shipped["detected"]]["episode_id"]) -
                  set(per_episode[(per_episode["washout"] == 1) &
                                  per_episode["detected"]]["episode_id"]))
    lines += ["",
              f"Short-horizon episodes the shipped washout misses that some "
              f"other washout recovers: **{len(gained)}** "
              f"({', '.join(gained) if gained else 'none'}).",
              f"Short-horizon episodes the shipped washout detects that "
              f"washout 1 loses: **{len(lost)}** "
              f"({', '.join(lost) if lost else 'none'}).", ""]
    lines += ["", "Per-step channel streams for every row above are in "
              "`results/tables/short_horizon_per_episode.csv` "
              "(`e_cusum_by_step`, `m_cusum_by_step`, `fused_cusum_by_step`, "
              "and the raw increments behind them).", ""]
    OUT_REPORT.write_text("\n".join(lines), "utf-8")
    print(f"[short-horizon] wrote {OUT_CSV}")
    print(f"[short-horizon] wrote {OUT_PER_EPISODE}")
    print(f"[short-horizon] wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
