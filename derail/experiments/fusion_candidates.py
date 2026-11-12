"""Which fused statistic, and does cross-fitting give a defensible threshold?

Development data only. The framework corpora were spent as evaluation data
when their interior was opened; they are used here to choose a design, and
the design is frozen before anything fresh is collected.

Four arms, all through the same code path so the comparison is of one thing
at a time:

  cusum / unnormalized   what ships today -- max over each channel's raw
                         CUSUM state, which is what lets the loudest channel
                         set the threshold;
  cusum / normalized     same statistic, each channel mapped through its own
                         healthy null first;
  rate / normalized      CUSUM divided by the steps scored so far, so a short
                         episode is judged on the same scale as a long one;
  increment / normalized no accumulation at all -- the step's own evidence.

Two threshold protocols, which is the outstanding sample-size question:

  holdout    disjoint fit / scale / threshold folds;
  crossfit   K-fold over the whole development pool, out-of-fold fused scores
             pooled to pick theta, then monitor and scales refitted on the
             entire pool. Cheaper in data, but the threshold is picked from
             scores produced by K different monitors and then applied to a
             final one that is none of them. Whether that transfers is
             measured here rather than assumed.

No failure label touches a fit, a scale, a threshold or a fold boundary.
Labels appear only when detection is finally counted.

Writes:
  results/tables/fusion_candidates.csv
  results/tables/fusion_length_strata.csv
  results/tables/fusion_channel_wins.csv
  results/fusion_candidates.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from derail.common import Standardizer, rng_for
from derail.evaluation.metrics import episode_auc, first_alarm
from derail.experiments.audit_healthy_admission import classify_episode
from derail.experiments.diagnose_framework_misses import FRAMEWORKS, load_corpus
from derail.monitor.esn import _WASHOUT
from derail.monitor.fusion_protocol import PROTOCOL, RECORD_PATH as PROTOCOL_RECORD
from derail.monitor.normalized_fusion import (ChannelScale,
                                              NormalizedFusionMonitor,
                                              pick_fused_threshold, scoreable)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "results" / "tables" / "fusion_candidates.csv"
OUT_STRATA = REPO_ROOT / "results" / "tables" / "fusion_length_strata.csv"
OUT_WINS = REPO_ROOT / "results" / "tables" / "fusion_channel_wins.csv"
OUT_REPLICATES = REPO_ROOT / "results" / "tables" / "fusion_arm_replicates.csv"
OUT_PAIRED = REPO_ROOT / "results" / "tables" / "fusion_arm_paired.csv"
OUT_REPORT = REPO_ROOT / "results" / "fusion_candidates.md"

CHANNELS = ("e", "m")
FA_BUDGET = 0.05
SPLIT_SEED = 5150
#: Held out of every fit, scale and threshold in this study. Not the final
#: evaluation set -- this corpus is development data -- but the only healthy
#: episodes any false-alarm number here is measured on.
DEV_TEST_FRACTION = 0.30
CROSSFIT_FOLDS = 5
ARMS = (("cusum", False), ("cusum", True), ("rate", True), ("increment", True))
#: Trajectory-length bands for the strata check. Chosen from the corpora's own
#: quartiles rather than round numbers, and applied identically to healthy and
#: injected episodes.
LENGTH_BANDS = ((4, 5), (6, 7), (8, 99))


def identity_scales(n: int) -> list[ChannelScale]:
    """Scales that change nothing -- the 'before normalization' arm."""
    return [ChannelScale(median=0.0, dispersion=1.0, n_steps=0)
            for _ in range(n)]


def build_monitor(fit_episodes, statistic, seed):
    standardizer = Standardizer().fit(fit_episodes)
    monitor = NormalizedFusionMonitor(standardizer, channels=CHANNELS,
                                      statistic=statistic, seed=seed)
    monitor.fit(fit_episodes)
    return monitor


def verified_scoreable(corpus: str) -> tuple[list, list, dict]:
    """(scoreable, unscoreable, injected, counts) under the frozen rule.

    Healthy episodes must pass their task's own verifier; unscoreable ones are
    kept and counted but never used to fit, scale or threshold.
    """
    episodes, meta = load_corpus(corpus)
    by_id = {ep.episode_id: ep for ep in episodes}
    verified, unscoreable, rejected = [], [], 0
    for entry in meta.values():
        if entry["failure_class"] is not None:
            continue
        ep = by_id[entry["episode_id"]]
        steps = [json.loads(line) for line in
                 (REPO_ROOT / "traces" / corpus / entry["file"])
                 .read_text("utf-8").splitlines() if line.strip()]
        row = classify_episode({**entry, "_corpus": corpus}, steps)
        if row["admission_status"] != "verified_healthy":
            rejected += 1
            continue
        if not scoreable(ep):
            unscoreable.append(ep)
            continue
        verified.append(ep)
    injected = [ep for ep in episodes if not ep.is_healthy]
    return verified, unscoreable, injected, {
        "rejected_healthy_attempts": rejected,
        "verified_unscoreable": len(unscoreable),
        "verified_scoreable": len(verified),
    }


def _rate_ci(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    ci = sps.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return k / n, float(ci.low), float(ci.high)


def holdout_threshold(pool, statistic, normalized, seed):
    """Disjoint fit / scale / threshold folds, 60/20/20 of the pool."""
    n_fit = int(round(0.6 * len(pool)))
    n_scale = int(round(0.2 * len(pool)))
    fit, scale, thr = (pool[:n_fit], pool[n_fit:n_fit + n_scale],
                       pool[n_fit + n_scale:])
    monitor = build_monitor(fit, statistic, seed)
    if normalized:
        monitor.fit_channel_scales(scale)
    else:
        monitor.scales = identity_scales(len(CHANNELS))
    theta, used, skipped = pick_fused_threshold(monitor, thr,
                                                fa_budget=FA_BUDGET)
    return monitor, theta, {"n_fit": len(fit), "n_scale": len(scale),
                            "n_threshold": used, "n_threshold_skipped": skipped}


def crossfit_threshold(pool, statistic, normalized, seed):
    """Out-of-fold fused scores pick theta; the monitor is then refit on all."""
    folds = [pool[i::CROSSFIT_FOLDS] for i in range(CROSSFIT_FOLDS)]
    oof_streams = []
    for k in range(CROSSFIT_FOLDS):
        held = folds[k]
        rest = [ep for j, f in enumerate(folds) if j != k for ep in f]
        n_fit = int(round(0.75 * len(rest)))
        monitor = build_monitor(rest[:n_fit], statistic, seed)
        if normalized:
            monitor.fit_channel_scales(rest[n_fit:])
        else:
            monitor.scales = identity_scales(len(CHANNELS))
        oof_streams += [monitor.score_stream(ep) for ep in held
                        if scoreable(ep)]
    from derail.evaluation.metrics import pick_threshold
    theta = float(pick_threshold(oof_streams, fa_budget=FA_BUDGET,
                                 warn_infeasible=False))

    n_fit = int(round(0.75 * len(pool)))
    final = build_monitor(pool[:n_fit], statistic, seed)
    if normalized:
        final.fit_channel_scales(pool[n_fit:])
    else:
        final.scales = identity_scales(len(CHANNELS))
    return final, theta, {"n_fit": n_fit, "n_scale": len(pool) - n_fit,
                          "n_threshold": len(oof_streams),
                          "n_threshold_skipped": 0}


def channel_wins(monitor, episodes) -> dict[str, float]:
    counts = dict.fromkeys(monitor.channels, 0)
    total = 0
    for ep in episodes:
        monitor.start_episode()
        for t in range(ep.T):
            monitor.score_step(ep.X[t])
            if t < _WASHOUT:
                continue
            counts[monitor.last_winning_channel] += 1
            total += 1
    return {c: (n / total if total else float("nan")) for c, n in counts.items()}


def evaluate(monitor, theta, scoreable_healthy, unscoreable_healthy,
             injected) -> tuple[dict, list[dict]]:
    """Detection and false alarms, with both healthy denominators.

    An unscoreable episode cannot alarm, so it is counted in the operational
    denominator (a deployment still runs it) and excluded from the scoreable
    one (the detector was never given a chance on it). Reporting only the
    first flatters the rate; reporting only the second hides the coverage gap.
    """
    per_episode = []
    fa_scoreable = 0
    for ep in scoreable_healthy:
        stream = monitor.score_stream(ep)
        alarm = first_alarm(stream, theta)
        fa_scoreable += alarm is not None
        per_episode.append({"episode_id": ep.episode_id, "is_healthy": True,
                            "T": ep.T, "failure_class": None,
                            "alarmed": alarm is not None,
                            "alarm_step": alarm, "tau": None})
    for ep in unscoreable_healthy:
        per_episode.append({"episode_id": ep.episode_id, "is_healthy": True,
                            "T": ep.T, "failure_class": None,
                            "alarmed": False, "alarm_step": None, "tau": None})

    detected, delays = 0, []
    scores = {}
    for ep in injected:
        stream = monitor.score_stream(ep)
        scores[ep.episode_id] = stream
        alarm = first_alarm(stream, theta)
        hit = alarm is not None and alarm >= ep.tau
        detected += hit
        if hit:
            delays.append(alarm - ep.tau)
        per_episode.append({"episode_id": ep.episode_id, "is_healthy": False,
                            "T": ep.T, "failure_class": ep.failure_class,
                            "alarmed": hit, "alarm_step": alarm,
                            "tau": ep.tau})

    n_scoreable = len(scoreable_healthy)
    n_operational = n_scoreable + len(unscoreable_healthy)
    for ep in scoreable_healthy:
        scores[ep.episode_id] = monitor.score_stream(ep)
    det, det_lo, det_hi = _rate_ci(detected, len(injected))
    fa_s, fa_s_lo, fa_s_hi = _rate_ci(fa_scoreable, n_scoreable)
    fa_o, _, _ = _rate_ci(fa_scoreable, n_operational)
    return {
        "theta": theta,
        "n_injected": len(injected), "n_detected": detected,
        "detection_rate": det, "detection_ci_lo": det_lo,
        "detection_ci_hi": det_hi,
        "n_healthy_scoreable": n_scoreable,
        "n_healthy_operational": n_operational,
        "n_false_alarms": fa_scoreable,
        "scoreable_fa_rate": fa_s, "scoreable_fa_ci_lo": fa_s_lo,
        "scoreable_fa_ci_hi": fa_s_hi,
        "operational_fa_rate": fa_o,
        "monitoring_coverage": n_scoreable / n_operational if n_operational else float("nan"),
        "median_delay": float(np.median(delays)) if delays else float("nan"),
        "episode_auc": episode_auc(scoreable_healthy + injected, scores),
    }, per_episode


def strata_rows(per_episode: list[dict], arm: str, protocol: str,
                framework: str) -> list[dict]:
    """Detection and false alarms by trajectory length.

    The whole point of changing the statistic is that the old one was
    length-coupled. A replacement that simply moves the bias to a different
    band would look fine in the aggregate, so the bands are reported.
    """
    df = pd.DataFrame(per_episode)
    rows = []
    for lo, hi in LENGTH_BANDS:
        band = df[(df["T"] >= lo) & (df["T"] <= hi)]
        healthy = band[band["is_healthy"]]
        inj = band[~band["is_healthy"]]
        rows.append({
            "framework": framework, "arm": arm, "protocol": protocol,
            "length_band": f"{lo}-{hi if hi < 99 else '+'}",
            "n_healthy": len(healthy),
            "fa_rate": (healthy["alarmed"].mean() if len(healthy) else float("nan")),
            "n_injected": len(inj),
            "detection_rate": (inj["alarmed"].mean() if len(inj) else float("nan")),
            "median_alarm_step": (float(inj.loc[inj["alarmed"], "alarm_step"].median())
                                  if inj["alarmed"].any() else float("nan")),
        })
    return rows


def prefix_length_profile(monitor, theta, episodes) -> list[dict]:
    """Fused score against how far into the episode it was produced.

    A length-decoupled statistic should not drift upward with prefix length
    on healthy runs; if it does, the bias has moved rather than gone.
    """
    rows = []
    for ep in episodes:
        stream = monitor.score_stream(ep)
        for t in range(_WASHOUT, ep.T):
            rows.append({"prefix_steps": t - _WASHOUT + 1,
                         "score": float(stream[t]), "theta": theta})
    return rows


def per_class_detection(per_episode: list[dict]) -> dict[str, str]:
    df = pd.DataFrame(per_episode)
    inj = df[~df["is_healthy"]]
    return {f"det_{cls}": f"{int(g['alarmed'].sum())}/{len(g)}"
            for cls, g in inj.groupby("failure_class")}


def step_latency(monitor, episodes) -> tuple[float, float]:
    """Per-step scoring latency in microseconds, p50 and p95.

    An accuracy gain bought by making the monitor substantially heavier is
    not a free gain, so the cost is measured in the same pass rather than
    argued about afterwards.
    """
    import time

    samples = []
    for ep in episodes:
        monitor.start_episode()
        for t in range(ep.T):
            t0 = time.perf_counter()
            monitor.score_step(ep.X[t])
            samples.append((time.perf_counter() - t0) * 1e6)
    return float(np.percentile(samples, 50)), float(np.percentile(samples, 95))


#: Repeated dev-splits for the arm comparison. A single split of this pool
#: leaves a threshold fold of 17-20 episodes, which the committed sample-size
#: curve puts at the edge of identifiability -- so one split cannot separate
#: the arms, and the project's own rule (vary the split, pool paired
#: differences) applies. The native pool cannot substitute for more data here:
#: it records `success = None`, so under the frozen admission rule it supplies
#: no verified-healthy episodes at all.
#: 30 is this project's floor for a replicated comparison, not its target:
#: an earlier study saw a detection gain at p = 0.004 over replicates 0-29
#: come back at p = 0.948 over 500-529. 40 clears the floor and splits into
#: two blocks of 20, so the per-block spread is reportable alongside the
#: pooled mean rather than hidden by it.
ARM_REPLICATES = 40
ARM_BLOCK = 20


def replicated_arms(pool_all, unscoreable, injected, framework, replicates):
    """Each arm on the same splits, so differences are paired.

    Only the holdout protocol is replicated. Cross-fitting needs no power to
    settle: its threshold misses by an order of magnitude on a single split,
    and the reason is structural rather than sampling noise.
    """
    rows = []
    for replicate in range(replicates):
        perm = rng_for(SPLIT_SEED, "arm-replicate", framework,
                       replicate).permutation(len(pool_all))
        shuffled = [pool_all[i] for i in perm]
        n_test = int(round(DEV_TEST_FRACTION * len(shuffled)))
        dev_test, pool = shuffled[:n_test], shuffled[n_test:]
        for statistic, normalized in ARMS:
            arm = f"{statistic}/{'normalized' if normalized else 'raw'}"
            monitor, theta, folds = holdout_threshold(pool, statistic,
                                                      normalized, replicate)
            metrics, per_episode = evaluate(monitor, theta, dev_test,
                                            unscoreable, injected)
            win = channel_wins(monitor, dev_test + injected)
            rows.append({"framework": framework, "replicate": replicate,
                         "arm": arm, "statistic": statistic,
                         "normalized": normalized, **folds, **metrics,
                         **{f"win_share_{c}": v for c, v in win.items()}})
        print(f"[fusion] {framework} replicate {replicate} done")
    return rows


def paired_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled per-arm means and paired differences against the shipped arm.

    Paired because every arm saw the same splits: the split-to-split spread
    dominates the arm-to-arm difference on a pool this size, so unpaired means
    would compare noise.
    """
    baseline_arm = "cusum/raw"
    out = []
    for (framework, arm), group in df.groupby(["framework", "arm"]):
        base = df[(df["framework"] == framework) &
                  (df["arm"] == baseline_arm)].set_index("replicate")
        this = group.set_index("replicate")
        shared = sorted(set(base.index) & set(this.index))
        row = {"framework": framework, "arm": arm, "replicates": len(shared)}
        for metric in ("detection_rate", "scoreable_fa_rate", "episode_auc",
                       "win_share_m", "step_latency_p50_us"):
            if metric not in this.columns:
                continue
            values = this.loc[shared, metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
            blocks = [values.loc[[r for r in shared
                                  if r // ARM_BLOCK == b]].mean()
                      for b in range(max(1, len(shared) // ARM_BLOCK))]
            row[f"{metric}_block_means"] = ";".join(f"{b:.3f}" for b in blocks)
            row[f"{metric}_block_spread"] = (float(max(blocks) - min(blocks))
                                             if len(blocks) > 1 else 0.0)
            if arm != baseline_arm:
                diff = values - base.loc[shared, metric].astype(float)
                row[f"{metric}_paired_delta"] = float(diff.mean())
                if diff.abs().sum() > 0:
                    row[f"{metric}_wilcoxon_p"] = float(
                        sps.wilcoxon(diff, zero_method="zsplit").pvalue)
                else:
                    row[f"{metric}_wilcoxon_p"] = 1.0
        out.append(row)
    frame = pd.DataFrame(out)
    for framework in frame["framework"].unique():
        mask = frame["framework"] == framework
        raw = {}
        for index, row in frame[mask].iterrows():
            for column in frame.columns:
                if column.endswith("_wilcoxon_p") and pd.notna(row[column]):
                    raw[(index, column)] = float(row[column])
        for (index, column), adjusted in holm(raw).items():
            frame.loc[index, column.replace("_wilcoxon_p", "_holm_p")] = adjusted
    return frame


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, so testing several metrics at once
    does not manufacture a winner."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    n = len(ordered)
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (n - rank) * p))
        adjusted[key] = running
    return adjusted


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from the committed tables "
                         "without refitting anything")
    args = ap.parse_args(argv)
    if args.report_only:
        write_report(pd.read_csv(OUT_CSV), pd.read_csv(OUT_STRATA),
                     pd.read_csv(OUT_WINS), None,
                     pd.read_csv(OUT_REPLICATES))
        print(f"[fusion] wrote {OUT_REPORT}")
        return

    rows, strata, wins, prefix, replicate_rows = [], [], [], [], []
    for corpus, framework in FRAMEWORKS.items():
        pool_all, unscoreable, injected, counts = verified_scoreable(corpus)
        replicate_rows += replicated_arms(pool_all, unscoreable, injected,
                                          framework, ARM_REPLICATES)
        perm = rng_for(SPLIT_SEED, "fusion-dev-split",
                       framework).permutation(len(pool_all))
        shuffled = [pool_all[i] for i in perm]
        n_test = int(round(DEV_TEST_FRACTION * len(shuffled)))
        dev_test, pool = shuffled[:n_test], shuffled[n_test:]
        print(f"[fusion] {framework}: verified scoreable "
              f"{counts['verified_scoreable']}, unscoreable "
              f"{counts['verified_unscoreable']}, rejected "
              f"{counts['rejected_healthy_attempts']}; pool {len(pool)}, "
              f"dev-test {len(dev_test)}, injected {len(injected)}")

        for statistic, normalized in ARMS:
            arm = f"{statistic}/{'normalized' if normalized else 'raw'}"
            for protocol, build in (("holdout", holdout_threshold),
                                    ("crossfit", crossfit_threshold)):
                monitor, theta, folds = build(pool, statistic, normalized, 0)
                metrics, per_episode = evaluate(monitor, theta, dev_test,
                                                unscoreable, injected)
                p50, p95 = step_latency(monitor, dev_test)
                rows.append({"framework": framework, "dataset": corpus,
                             "arm": arm, "statistic": statistic,
                             "normalized": normalized, "protocol": protocol,
                             **folds, **metrics,
                             **per_class_detection(per_episode),
                             "step_latency_p50_us": p50,
                             "step_latency_p95_us": p95})
                strata += strata_rows(per_episode, arm, protocol, framework)
                win = channel_wins(monitor, dev_test + injected)
                wins.append({"framework": framework, "arm": arm,
                             "protocol": protocol,
                             **{f"win_share_{c}": v for c, v in win.items()}})
                if protocol == "holdout":
                    prefix += [{"framework": framework, "arm": arm, **r}
                               for r in prefix_length_profile(monitor, theta,
                                                              dev_test)]
                print(f"[fusion]   {arm:24s} {protocol:9s} theta={theta:9.3f} "
                      f"det={metrics['detection_rate']:.3f} "
                      f"fa_s={metrics['scoreable_fa_rate']:.3f} "
                      f"auc={metrics['episode_auc']:.3f}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    pd.DataFrame(replicate_rows).to_csv(OUT_REPLICATES, index=False)
    paired_summary(pd.DataFrame(replicate_rows)).to_csv(OUT_PAIRED, index=False)
    pd.DataFrame(strata).to_csv(OUT_STRATA, index=False)
    pd.DataFrame(wins).to_csv(OUT_WINS, index=False)
    write_report(df, pd.DataFrame(strata), pd.DataFrame(wins),
                 pd.DataFrame(prefix), pd.DataFrame(replicate_rows))
    for path in (OUT_CSV, OUT_REPLICATES, OUT_PAIRED, OUT_STRATA, OUT_WINS,
                 OUT_REPORT):
        print(f"[fusion] wrote {path}")


def write_report(df, strata, wins, prefix, replicates) -> None:
    lines = ["# Fusing on a common healthy-null scale",
             "",
             "**SUPPORTING / diagnostic evidence, development data only.** "
             "These corpora were spent when their interior was opened; they "
             "choose a design here and the design is frozen before anything "
             "fresh is collected. No number below is a performance claim, and "
             "the pre-fix baseline "
             "(`results/framework_generalized_monitor_report.md`) stands "
             "unchanged.",
             "",
             "Healthy episodes are admitted by the frozen rule: the task's own "
             "verifier must pass. Unscoreable verified-healthy episodes are "
             "never used to fit, scale or threshold, and never enter a "
             "threshold distribution as zeros; they are counted in the "
             "operational false-alarm denominator, because a deployment still "
             "runs them.",
             "",
             "`scoreable FA` divides by the healthy episodes the detector "
             "could act on; `operational FA` divides by every verified-healthy "
             "episode. They differ exactly by the monitoring coverage.",
             "",
             "| framework | arm | protocol | fit / scale / threshold | theta | "
             "detection (95% CI) | scoreable FA (95% CI) | operational FA | "
             "AUC | p50 us/step |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['framework']} | {r['arm']} | {r['protocol']} | "
            f"{int(r['n_fit'])} / {int(r['n_scale'])} / "
            f"{int(r['n_threshold'])} | {r['theta']:.3f} | "
            f"{r['detection_rate']:.3f} ({r['detection_ci_lo']:.3f}-"
            f"{r['detection_ci_hi']:.3f}) | {r['scoreable_fa_rate']:.3f} "
            f"({r['scoreable_fa_ci_lo']:.3f}-{r['scoreable_fa_ci_hi']:.3f}) | "
            f"{r['operational_fa_rate']:.3f} | {r['episode_auc']:.3f} | "
            f"{r['step_latency_p50_us']:.1f} |")

    paired = paired_summary(replicates)
    lines += ["", f"## Arm comparison over {ARM_REPLICATES} dev-splits", "",
              "A single split of this pool leaves a threshold fold of 17-20 "
              "episodes, which the committed sample-size curve puts at the "
              "edge of identifiability, so one split cannot separate the "
              "arms. Every arm sees the same splits and differences are "
              "paired against the shipped `cusum/raw`; Wilcoxon is over "
              "splits. The native corpora cannot supply more data here: they "
              "record `success = None`, so the frozen admission rule admits "
              "none of them as verified healthy.",
              "",
              "| framework | arm | detection (sd) | block means | "
              "d vs shipped | Wilcoxon p | Holm p | scoreable FA | d vs "
              "shipped | Holm p | m win share |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in paired.iterrows():
        def cell(metric):
            delta = r.get(f"{metric}_paired_delta")
            pval = r.get(f"{metric}_wilcoxon_p")
            if delta is None or pd.isna(delta):
                return "--", "--"
            return f"{delta:+.3f}", f"{pval:.3f}"
        det_d, det_p = cell("detection_rate")
        fa_d, _ = cell("scoreable_fa_rate")
        det_holm = r.get("detection_rate_holm_p")
        fa_holm = r.get("scoreable_fa_rate_holm_p")
        lines.append(
            f"| {r['framework']} | {r['arm']} | "
            f"{r['detection_rate_mean']:.3f} ({r['detection_rate_sd']:.3f}) | "
            f"{r['detection_rate_block_means']} | {det_d} | {det_p} | "
            f"{'--' if pd.isna(det_holm) else f'{det_holm:.3f}'} | "
            f"{r['scoreable_fa_rate_mean']:.3f} | {fa_d} | "
            f"{'--' if pd.isna(fa_holm) else f'{fa_holm:.3f}'} | "
            f"{r['win_share_m_mean']:.3f} |")

    lines += ["", "## Which channel wins the fusion", "",
              "The defect this addresses: on the shipped design the semantic "
              "channel wins almost every step because its CUSUM is simply "
              "larger, not because it is better informed.",
              "",
              "| framework | arm | protocol | e wins | m wins |",
              "|---|---|---|---|---|"]
    for _, r in wins.iterrows():
        lines.append(f"| {r['framework']} | {r['arm']} | {r['protocol']} | "
                     f"{r['win_share_e']:.3f} | {r['win_share_m']:.3f} |")

    lines += ["", "## Length strata", "",
              "A statistic that merely relocated the length bias would look "
              "fine in the aggregate. Detection and false alarms are reported "
              "per trajectory-length band, with the median alarm step.",
              "",
              "| framework | arm | protocol | band | n healthy | FA | "
              "n injected | detection | median alarm step |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in strata.iterrows():
        lines.append(
            f"| {r['framework']} | {r['arm']} | {r['protocol']} | "
            f"{r['length_band']} | {int(r['n_healthy'])} | {r['fa_rate']:.3f} | "
            f"{int(r['n_injected'])} | {r['detection_rate']:.3f} | "
            f"{r['median_alarm_step']:.1f} |")

    lines += ["", "## The protocol this selects", "",
              f"Recorded as a checked artifact in "
              f"`{PROTOCOL_RECORD.relative_to(REPO_ROOT).as_posix()}` and "
              f"pinned by a test, so the choice cannot drift from the code "
              f"that implements it.",
              "",
              f"- statistic **{PROTOCOL.statistic}**: the only arm positive "
              f"on both deployments, significant on one after Holm "
              f"correction, and the one that flattens the false-alarm "
              f"gradient across trajectory-length bands;",
              f"- per-channel normalization **on**, but not because it "
              f"detects more -- it does not, on either deployment. It is "
              f"what stops the fusion being decided by which channel carries "
              f"larger numbers;",
              f"- threshold on the **fused** statistic at a "
              f"{PROTOCOL.fa_budget:.0%} budget, never inferred from "
              f"per-channel budgets;",
              f"- **{PROTOCOL.protocol}**: fit >= "
              f"{PROTOCOL.min_fit_episodes}, scale >= "
              f"{PROTOCOL.min_scale_episodes}, threshold >= "
              f"{PROTOCOL.min_threshold_episodes}, untouched test "
              f"{PROTOCOL.untouched_test_episodes} per deployment. "
              f"Cross-fitting was measured and rejected.",
              "",
              "## Healthy score against prefix length", "",
              "Mean fused score on held-out healthy episodes by how many "
              "steps have been scored. A statistic that has decoupled length "
              "should be flat here; one that has not will climb.",
              "",
              "| framework | arm | prefix steps | mean score | n |",
              "|---|---|---|---|---|"]
    if prefix is None or prefix.empty:
        lines.append("")
        OUT_REPORT.write_text("\n".join(lines), "utf-8")
        return
    grouped = prefix.groupby(["framework", "arm", "prefix_steps"]).agg(
        mean_score=("score", "mean"), n=("score", "size")).reset_index()
    for _, r in grouped[grouped["prefix_steps"] <= 6].iterrows():
        lines.append(f"| {r['framework']} | {r['arm']} | "
                     f"{int(r['prefix_steps'])} | {r['mean_score']:.3f} | "
                     f"{int(r['n'])} |")
    lines.append("")
    OUT_REPORT.write_text("\n".join(lines), "utf-8")


if __name__ == "__main__":
    main()
