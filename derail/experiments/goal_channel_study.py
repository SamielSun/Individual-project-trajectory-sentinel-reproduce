"""Does a goal-consistency channel expose the drift the behavioural ones miss?

Development data only. The framework corpora were spent when their interior
was opened; they choose a design here, and the design is frozen before
anything fresh is collected.

The question is narrow on purpose. `normalized_fusion[e+m]/rate/v2` is frozen:
the statistic, the normalization, the three-fold holdout protocol and the 5%
budget are not reopened. The only thing varied is which channels are fused, so
any difference is attributable to the channel and not to a recalibration.

Three arms:

  e+m      the frozen protocol as it stands;
  e+m+c    the same, with the goal-consistency channel added to the fusion;
  c        the goal channel alone -- not a candidate for deployment, but the
           only way to see what it contributes independently of whether the
           max happens to select it.

Everything about the channel goes through the same healthy-only path as the
others: its scale is learned from the healthy null the other channels' scales
come from, and the threshold is picked on the fused statistic. It gets no
threshold of its own and no special case anywhere.

Two corpora families, for two different jobs:

  framework    langgraph7b_real2 / autogen7b_real2. Healthy episodes admitted
               by the frozen rule (the task's own verifier must pass), so the
               false-alarm numbers mean what they say. Six goal_drift episodes
               each -- enough to see a mechanism, not enough to measure a rate.
  long-drift   real_research7b_long_drift, 120 injected and 120 healthy on one
               long task. The only corpus with enough goal_drift to say
               anything statistical. Its manifest records no task name, so no
               registered verifier applies and its healthy pool is admitted by
               the collection rule instead. Its false-alarm figures are
               therefore NOT comparable with the framework ones and are
               reported separately, never pooled.

goal_drift detection is reported against two denominators, both of them:
every injected episode, and only those the realization audit found actually
went off task. Neither alone is honest -- see
`results/drift_realization_audit.md`.

Writes:
  results/tables/goal_channel_arms.csv
  results/tables/goal_channel_per_class.csv
  results/tables/goal_channel_paired.csv
  results/tables/goal_channel_strata.csv
  results/tables/goal_channel_threshold_allocation.csv
  results/goal_channel_study.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from derail.common import Standardizer, rng_for
from derail.evaluation.metrics import first_alarm
from derail.experiments.audit_drift_realization import classify_realization
from derail.experiments.audit_healthy_admission import classify_episode
from derail.experiments.fusion_candidates import (FA_BUDGET, LENGTH_BANDS,
                                                  SPLIT_SEED, _rate_ci, holm)
from derail.monitor.esn import _WASHOUT
from derail.monitor.fusion_protocol import PROTOCOL
from derail.monitor.normalized_fusion import (NormalizedFusionMonitor,
                                              pick_fused_threshold, scoreable)
from derail.telemetry.adapter import episode_from_trace

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_ROOT = REPO_ROOT / "traces"
OUT_ARMS = REPO_ROOT / "results" / "tables" / "goal_channel_arms.csv"
OUT_PER_CLASS = REPO_ROOT / "results" / "tables" / "goal_channel_per_class.csv"
OUT_PAIRED = REPO_ROOT / "results" / "tables" / "goal_channel_paired.csv"
OUT_STRATA = REPO_ROOT / "results" / "tables" / "goal_channel_strata.csv"
OUT_ALLOCATION = (REPO_ROOT / "results" / "tables" /
                  "goal_channel_threshold_allocation.csv")
OUT_COST = REPO_ROOT / "results" / "tables" / "goal_channel_cost.csv"
OUT_REPORT = REPO_ROOT / "results" / "goal_channel_study.md"

#: Which channels each arm fuses. The first is the frozen protocol and is the
#: baseline every paired difference is taken against.
ARMS: tuple[tuple[str, ...], ...] = (("e", "m"), ("e", "m", "c"), ("c",))
BASELINE_ARM = "e+m"

FRAMEWORK_CORPORA = ("langgraph7b_real2", "autogen7b_real2")
LONG_CORPUS = "real_research7b_long_drift"
CORPORA = FRAMEWORK_CORPORA + (LONG_CORPUS,)

DEV_TEST_FRACTION = 0.30
#: Past this project's replicate floor of 30, in two blocks, because a gain
#: measured at 20 splits has previously moved materially by 40.
REPLICATES = 40
BLOCK = 20


def arm_name(channels: tuple[str, ...]) -> str:
    return "+".join(channels)


def load_goal_corpus(corpus: str, reliability: bool = False) -> tuple[list, dict]:
    """Episodes carrying the goal channel, plus their manifest entries."""
    corpus_dir = TRACES_ROOT / corpus
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    episodes, meta = [], {}
    for entry in manifest:
        steps = [json.loads(line) for line in
                 (corpus_dir / entry["file"]).read_text("utf-8").splitlines()
                 if line.strip()]
        episodes.append(episode_from_trace(
            steps, entry["episode_id"],
            tau=entry["tau"] if entry["failure_class"] else None,
            failure_class=entry["failure_class"],
            use_sentence_transformers=False, goal=True,
            reliability=reliability))
        meta[entry["episode_id"]] = {**entry, "_steps": steps}
    return episodes, meta


def admit(corpus: str, episodes: list, meta: dict) -> tuple[list, list, list, dict]:
    """(scoreable healthy, unscoreable healthy, injected, counts).

    The frozen admission rule needs a registered verifier. Where the corpus
    records no task name none applies, and that is reported rather than
    papered over: those healthy episodes enter on the collection rule, and
    every false-alarm number derived from them is labelled accordingly.
    """
    by_id = {ep.episode_id: ep for ep in episodes}
    verifiable = all(entry.get("task") for entry in meta.values())
    verified, unscoreable, rejected = [], [], 0
    for entry in meta.values():
        if entry["failure_class"] is not None:
            continue
        ep = by_id[entry["episode_id"]]
        if verifiable:
            row = classify_episode({**entry, "_corpus": corpus}, entry["_steps"])
            if row["admission_status"] != "verified_healthy":
                rejected += 1
                continue
        (verified if scoreable(ep) else unscoreable).append(ep)
    injected = [ep for ep in episodes if not ep.is_healthy]
    return verified, unscoreable, injected, {
        "healthy_admission": ("task_verifier_must_pass" if verifiable
                              else "collection_rule_only__no_registered_verifier"),
        "rejected_healthy_attempts": rejected,
        "verified_unscoreable": len(unscoreable),
        "verified_scoreable": len(verified),
    }


def realization_map(corpus: str, meta: dict) -> dict[str, str]:
    """episode_id -> realization, for injected goal_drift only."""
    out = {}
    for entry in meta.values():
        if entry.get("failure_class") != "goal_drift":
            continue
        out[entry["episode_id"]] = classify_realization(
            entry["_steps"], entry.get("tau"))["realization"]
    return out


def build(pool: list, channels: tuple[str, ...], seed: int):
    """Three disjoint healthy folds, exactly as the frozen protocol says."""
    n_fit = int(round(0.6 * len(pool)))
    n_scale = int(round(0.2 * len(pool)))
    fit, scale, thr = (pool[:n_fit], pool[n_fit:n_fit + n_scale],
                       pool[n_fit + n_scale:])
    monitor = NormalizedFusionMonitor(Standardizer().fit(fit), channels=channels,
                                      statistic=PROTOCOL.statistic, seed=seed)
    monitor.fit(fit)
    monitor.fit_channel_scales(scale)
    theta, used, skipped = pick_fused_threshold(monitor, thr,
                                                fa_budget=FA_BUDGET)
    return monitor, theta, {"n_fit": len(fit), "n_scale": len(scale),
                            "n_threshold": used, "n_threshold_skipped": skipped}


def evaluate(monitor, theta, healthy, unscoreable, injected,
             realization: dict[str, str]) -> tuple[dict, list[dict]]:
    per_episode, false_alarms = [], 0
    for ep in healthy:
        alarm = first_alarm(monitor.score_stream(ep), theta)
        false_alarms += alarm is not None
        per_episode.append({"episode_id": ep.episode_id, "is_healthy": True,
                            "failure_class": None, "realization": None,
                            "T": ep.T, "alarmed": alarm is not None,
                            "alarm_step": alarm, "tau": None})
    for ep in unscoreable:
        per_episode.append({"episode_id": ep.episode_id, "is_healthy": True,
                            "failure_class": None, "realization": None,
                            "T": ep.T, "alarmed": False, "alarm_step": None,
                            "tau": None})
    detected = 0
    for ep in injected:
        alarm = first_alarm(monitor.score_stream(ep), theta)
        hit = alarm is not None and alarm >= ep.tau
        detected += hit
        per_episode.append({"episode_id": ep.episode_id, "is_healthy": False,
                            "failure_class": ep.failure_class,
                            "realization": realization.get(ep.episode_id),
                            "T": ep.T, "alarmed": hit, "alarm_step": alarm,
                            "tau": ep.tau})

    n_scoreable, n_operational = len(healthy), len(healthy) + len(unscoreable)
    det, det_lo, det_hi = _rate_ci(detected, len(injected))
    fa, fa_lo, fa_hi = _rate_ci(false_alarms, n_scoreable)
    metrics = {
        "theta": theta, "n_injected": len(injected), "n_detected": detected,
        "detection_rate": det, "detection_ci_lo": det_lo,
        "detection_ci_hi": det_hi,
        "n_healthy_scoreable": n_scoreable,
        "n_healthy_operational": n_operational,
        "n_false_alarms": false_alarms,
        "scoreable_fa_rate": fa, "scoreable_fa_ci_lo": fa_lo,
        "scoreable_fa_ci_hi": fa_hi,
        "operational_fa_rate": _rate_ci(false_alarms, n_operational)[0],
    }
    metrics.update(drift_denominators(per_episode))
    return metrics, per_episode


def drift_denominators(per_episode: list[dict]) -> dict:
    """goal_drift detection against both denominators, and the alarms raised
    on hijacks the agent ignored -- which are false alarms wearing an injected
    label, and are counted as such rather than as detections."""
    rows = [r for r in per_episode if r["failure_class"] == "goal_drift"]
    realized = [r for r in rows if r["realization"] != "not_realized"]
    ignored = [r for r in rows if r["realization"] == "not_realized"]
    hit_all = sum(r["alarmed"] for r in rows)
    hit_real = sum(r["alarmed"] for r in realized)
    return {
        "drift_n_injected": len(rows), "drift_detected_injected": hit_all,
        "drift_det_injected": hit_all / len(rows) if rows else float("nan"),
        "drift_n_realized": len(realized), "drift_detected_realized": hit_real,
        "drift_det_realized": (hit_real / len(realized) if realized
                               else float("nan")),
        "drift_n_ignored": len(ignored),
        "drift_alarms_on_ignored": sum(r["alarmed"] for r in ignored),
    }


def per_class_rows(per_episode: list[dict]) -> list[dict]:
    out = []
    injected = [r for r in per_episode if not r["is_healthy"]]
    for failure_class in sorted({r["failure_class"] for r in injected}):
        group = [r for r in injected if r["failure_class"] == failure_class]
        hits = sum(r["alarmed"] for r in group)
        rate, lo, hi = _rate_ci(hits, len(group))
        out.append({"failure_class": failure_class, "n": len(group),
                    "detected": hits, "detection_rate": rate,
                    "ci_lo": lo, "ci_hi": hi})
    return out


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


def strata_rows(per_episode: list[dict], arm: str, corpus: str) -> list[dict]:
    """Detection and false alarms by trajectory length, so a gain that is
    really a length effect cannot hide in the aggregate."""
    out = []
    for lo, hi in LENGTH_BANDS:
        band = [r for r in per_episode if lo <= r["T"] <= hi]
        healthy = [r for r in band if r["is_healthy"]]
        injected = [r for r in band if not r["is_healthy"]]
        alarms = [r["alarm_step"] for r in injected
                  if r["alarmed"] and r["alarm_step"] is not None]
        out.append({
            "dataset": corpus, "arm": arm, "length_band": f"{lo}-{hi}",
            "n_healthy": len(healthy),
            "fa_rate": (sum(r["alarmed"] for r in healthy) / len(healthy)
                        if healthy else float("nan")),
            "n_injected": len(injected),
            "detection_rate": (sum(r["alarmed"] for r in injected) /
                               len(injected) if injected else float("nan")),
            "median_alarm_step": float(np.median(alarms)) if alarms else float("nan"),
        })
    return out


def step_latency(monitor, episodes) -> tuple[float, float]:
    samples = []
    for ep in episodes:
        monitor.start_episode()
        for t in range(ep.T):
            start = time.perf_counter()
            monitor.score_step(ep.X[t])
            samples.append((time.perf_counter() - start) * 1e6)
    return (float(np.percentile(samples, 50)),
            float(np.percentile(samples, 95)))


def telemetry_cost(corpus: str, n_episodes: int = 25) -> dict:
    """What the channel costs to PRODUCE, and what it costs to carry.

    The per-step scoring figures elsewhere measure the monitor only. A channel
    also has to be computed from the raw step before any monitor sees it, and
    it has to be held in memory per episode -- both are real deployment costs
    and neither shows up in a scoring benchmark.
    """
    corpus_dir = TRACES_ROOT / corpus
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    traces = []
    for entry in manifest[:n_episodes]:
        traces.append((entry["episode_id"], [
            json.loads(line) for line in
            (corpus_dir / entry["file"]).read_text("utf-8").splitlines()
            if line.strip()]))

    row = {"dataset": corpus, "n_episodes": len(traces),
           "n_steps": sum(len(s) for _, s in traces)}
    for label, kwargs in (("v4_grounding", {"grounding": True}),
                          ("v5_goal", {"goal": True})):
        start = time.perf_counter()
        episodes = [episode_from_trace(steps, episode_id,
                                       use_sentence_transformers=False,
                                       **kwargs)
                    for episode_id, steps in traces]
        elapsed = time.perf_counter() - start
        row[f"{label}_build_us_per_step"] = elapsed * 1e6 / row["n_steps"]
        row[f"{label}_feature_bytes_per_step"] = float(
            sum(ep.X.nbytes for ep in episodes) / row["n_steps"])
        row[f"{label}_dims"] = int(episodes[0].X.shape[1])
    return row


def split(pool_all: list, seed_key: str, corpus: str, replicate: int):
    perm = rng_for(SPLIT_SEED, seed_key, corpus, replicate).permutation(len(pool_all))
    shuffled = [pool_all[i] for i in perm]
    n_test = int(round(DEV_TEST_FRACTION * len(shuffled)))
    return shuffled[:n_test], shuffled[n_test:]


def run_corpus(corpus: str) -> tuple[list[dict], list[dict], list[dict], dict]:
    episodes, meta = load_goal_corpus(corpus)
    healthy_all, unscoreable, injected, counts = admit(corpus, episodes, meta)
    realization = realization_map(corpus, meta)
    print(f"[goal-channel] {corpus}: healthy scoreable {len(healthy_all)}, "
          f"unscoreable {len(unscoreable)}, injected {len(injected)}, "
          f"admission {counts['healthy_admission']}")

    arms, per_class, strata = [], [], []
    for replicate in range(REPLICATES):
        dev_test, pool = split(healthy_all, "goal-arm", corpus, replicate)
        for channels in ARMS:
            name = arm_name(channels)
            monitor, theta, folds = build(pool, channels, replicate)
            metrics, per_episode = evaluate(monitor, theta, dev_test,
                                            unscoreable, injected, realization)
            wins = channel_wins(monitor, dev_test + injected)
            row = {"dataset": corpus, "arm": name, "replicate": replicate,
                   **counts, **folds, **metrics,
                   **{f"win_share_{c}": v for c, v in wins.items()}}
            if replicate == 0:
                p50, p95 = step_latency(monitor, dev_test)
                row["step_latency_p50_us"] = p50
                row["step_latency_p95_us"] = p95
                strata += strata_rows(per_episode, name, corpus)
            arms.append(row)
            per_class += [{"dataset": corpus, "arm": name,
                           "replicate": replicate, **r}
                          for r in per_class_rows(per_episode)]
        print(f"[goal-channel] {corpus} replicate {replicate} done")
    return arms, per_class, strata, counts


def threshold_allocation(corpus: str, healthy: list, unscoreable: list,
                         injected: list, realization: dict,
                         replicates: int = 15) -> list[dict]:
    """Is a loss on the other classes masking, or budget re-allocation?

    The three-channel monitor is scored twice: at its own threshold, and at
    the two-channel monitor's. If the second restores the other classes, the
    goal channel hides nothing -- the fixed false-alarm budget is simply being
    divided across one more channel, and the extra detection is paid for in
    budget rather than bought for free.

    The third row is what that costs, so the trade is visible rather than
    argued about.
    """
    rows = []
    for replicate in range(replicates):
        dev_test, pool = split(healthy, "goal-arm", corpus, replicate)
        pair, theta_pair, _ = build(pool, ("e", "m"), replicate)
        trio, theta_trio, _ = build(pool, ("e", "m", "c"), replicate)
        for label, monitor, theta in (
                ("e+m at its own theta", pair, theta_pair),
                ("e+m+c at its own theta", trio, theta_trio),
                ("e+m+c at the e+m theta", trio, theta_pair)):
            metrics, per_episode = evaluate(monitor, theta, dev_test,
                                            unscoreable, injected, realization)
            rows.append({"dataset": corpus, "condition": label,
                         "replicate": replicate,
                         "theta_ratio": theta_trio / theta_pair,
                         **{f"det_{r['failure_class']}": r["detection_rate"]
                            for r in per_class_rows(per_episode)},
                         "scoreable_fa_rate": metrics["scoreable_fa_rate"],
                         "drift_det_realized": metrics["drift_det_realized"]})
    return rows


def paired(df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm means and paired differences against the frozen protocol.

    Paired because every arm saw the same splits: on pools this size the
    split-to-split spread dominates the arm-to-arm difference, so unpaired
    means would compare noise.
    """
    metrics = ("detection_rate", "scoreable_fa_rate", "drift_det_injected",
               "drift_det_realized")
    out = []
    for (corpus, arm), group in df.groupby(["dataset", "arm"]):
        base = df[(df["dataset"] == corpus) &
                  (df["arm"] == BASELINE_ARM)].set_index("replicate")
        this = group.set_index("replicate")
        shared = sorted(set(base.index) & set(this.index))
        row = {"dataset": corpus, "arm": arm, "replicates": len(shared)}
        for metric in metrics:
            values = this.loc[shared, metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
            blocks = [float(values.iloc[i:i + BLOCK].mean())
                      for i in range(0, len(values), BLOCK)]
            row[f"{metric}_blocks"] = ";".join(f"{b:.3f}" for b in blocks)
            if arm == BASELINE_ARM:
                continue
            diff = (values - base.loc[shared, metric].astype(float)).dropna()
            row[f"{metric}_paired_delta"] = float(diff.mean())
            row[f"{metric}_wilcoxon_p"] = (
                float(sps.wilcoxon(diff, zero_method="zsplit").pvalue)
                if len(diff) and diff.abs().sum() > 0 else 1.0)
        out.append(row)

    result = pd.DataFrame(out)
    for metric in metrics:
        column = f"{metric}_wilcoxon_p"
        if column not in result.columns:
            continue
        for corpus, group in result.groupby("dataset"):
            valid = group[column].notna()
            adjusted = holm({int(i): float(group.loc[i, column])
                             for i in group.index[valid]})
            for index, value in adjusted.items():
                result.loc[index, f"{metric}_holm_p"] = value
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.goal_channel_study")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild the report from committed tables")
    args = parser.parse_args(argv)

    if args.report_only:
        arms = pd.read_csv(OUT_ARMS)
        per_class = pd.read_csv(OUT_PER_CLASS)
        strata = pd.read_csv(OUT_STRATA)
        paired_df = pd.read_csv(OUT_PAIRED)
    else:
        all_arms, all_class, all_strata, all_alloc = [], [], [], []
        for corpus in CORPORA:
            a, c, s, _ = run_corpus(corpus)
            all_arms += a
            all_class += c
            all_strata += s
            episodes, meta = load_goal_corpus(corpus)
            healthy, unscoreable, injected, _ = admit(corpus, episodes, meta)
            all_alloc += threshold_allocation(
                corpus, healthy, unscoreable, injected,
                realization_map(corpus, meta))
        pd.DataFrame(all_alloc).to_csv(OUT_ALLOCATION, index=False)
        arms = pd.DataFrame(all_arms)
        per_class = pd.DataFrame(all_class)
        strata = pd.DataFrame(all_strata)
        paired_df = paired(arms)
        OUT_ARMS.parent.mkdir(parents=True, exist_ok=True)
        arms.to_csv(OUT_ARMS, index=False)
        per_class.to_csv(OUT_PER_CLASS, index=False)
        strata.to_csv(OUT_STRATA, index=False)
        paired_df.to_csv(OUT_PAIRED, index=False)

    allocation = pd.read_csv(OUT_ALLOCATION)
    cost = pd.DataFrame([telemetry_cost(c) for c in CORPORA])
    cost.to_csv(OUT_COST, index=False)
    write_report(arms, per_class, strata, paired_df, allocation, cost)
    for path in (OUT_ARMS, OUT_PER_CLASS, OUT_PAIRED, OUT_STRATA,
                 OUT_ALLOCATION, OUT_COST, OUT_REPORT):
        print(f"[goal-channel] wrote {path}")


def write_report(arms, per_class, strata, paired_df, allocation,
                 cost) -> None:
    lines = [
        "# A goal-consistency channel for drift the behavioural channels miss",
        "",
        "**SUPPORTING / diagnostic evidence, development data only.** These "
        "corpora were spent when their interior was opened. Nothing below is "
        "a performance claim, and the pre-fix baseline "
        "(`results/framework_generalized_monitor_report.md`) stands "
        "unchanged.",
        "",
        "`normalized_fusion[e+m]/rate/v2` is held fixed: statistic, "
        "normalization, three-fold holdout and the 5% budget are not "
        "reopened. Only the fused channel set varies, so a difference is "
        "attributable to the channel rather than to a recalibration. The goal "
        "channel is scaled and thresholded through the same healthy-only path "
        "as the others and has no threshold of its own.",
        "",
        f"Every figure is the mean of {REPLICATES} paired development splits, "
        f"reported with its two blocks of {BLOCK} so a result that moved "
        "between blocks cannot be read as settled.",
        "",
        "## Arms",
        "",
        "| dataset | arm | admission | detection | scoreable FA | "
        "goal_drift (injected) | goal_drift (realized) | alarms on ignored "
        "hijacks |",
        "|---|---|---|---|---|---|---|---|",
    ]
    grouped = arms.groupby(["dataset", "arm"], sort=False)
    for (corpus, arm), group in grouped:
        lines.append(
            f"| {corpus} | {arm} | "
            f"{group['healthy_admission'].iloc[0].replace('_', ' ')} | "
            f"{group['detection_rate'].mean():.3f} | "
            f"{group['scoreable_fa_rate'].mean():.3f} | "
            f"{group['drift_detected_injected'].mean():.2f}/"
            f"{group['drift_n_injected'].iloc[0]:.0f} = "
            f"{group['drift_det_injected'].mean():.3f} | "
            f"{group['drift_detected_realized'].mean():.2f}/"
            f"{group['drift_n_realized'].iloc[0]:.0f} = "
            f"{group['drift_det_realized'].mean():.3f} | "
            f"{group['drift_alarms_on_ignored'].mean():.2f}/"
            f"{group['drift_n_ignored'].iloc[0]:.0f} |")

    lines += ["", "## Paired against the frozen protocol", "",
              "Holm-corrected within each deployment, because several metrics "
              "are tested per deployment.",
              "",
              "| dataset | arm | metric | mean | delta | blocks | Wilcoxon p | "
              "Holm p |", "|---|---|---|---|---|---|---|---|"]
    for _, r in paired_df.iterrows():
        if r["arm"] == BASELINE_ARM:
            continue
        for metric in ("detection_rate", "scoreable_fa_rate",
                       "drift_det_realized"):
            delta = r.get(f"{metric}_paired_delta")
            if pd.isna(delta):
                continue
            lines.append(
                f"| {r['dataset']} | {r['arm']} | {metric} | "
                f"{r[f'{metric}_mean']:.3f} | {delta:+.3f} | "
                f"{r[f'{metric}_blocks']} | "
                f"{r[f'{metric}_wilcoxon_p']:.4f} | "
                f"{r.get(f'{metric}_holm_p', float('nan')):.4f} |")

    lines += ["", "## Per failure class", "",
              "A channel that bought goal_drift by raising the threshold "
              "would show it here as a loss on the other classes.",
              "",
              "| dataset | arm | class | n | detection |",
              "|---|---|---|---|---|"]
    pc = per_class.groupby(["dataset", "arm", "failure_class"], sort=False).agg(
        n=("n", "first"), detection=("detection_rate", "mean")).reset_index()
    for _, r in pc.iterrows():
        lines.append(f"| {r['dataset']} | {r['arm']} | {r['failure_class']} | "
                     f"{int(r['n'])} | {r['detection']:.3f} |")

    lines += ["", "## Which channel wins the fusion", "",
              "The goal channel is only useful if the max actually selects it "
              "when it is the informed one.",
              "", "| dataset | arm | e | m | c |", "|---|---|---|---|---|"]
    for (corpus, arm), group in grouped:
        cells = [f"{group[f'win_share_{c}'].mean():.3f}"
                 if f"win_share_{c}" in group.columns and
                 group[f"win_share_{c}"].notna().any() else "-"
                 for c in ("e", "m", "c")]
        lines.append(f"| {corpus} | {arm} | " + " | ".join(cells) + " |")

    lines += ["", "## Length strata", "",
              "| dataset | arm | band | n healthy | FA | n injected | "
              "detection | median alarm step |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in strata.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['arm']} | {r['length_band']} | "
            f"{int(r['n_healthy'])} | {r['fa_rate']:.3f} | "
            f"{int(r['n_injected'])} | {r['detection_rate']:.3f} | "
            f"{r['median_alarm_step']:.1f} |")

    lines += ["", "## Is the loss on the other classes masking, or budget?",
              "",
              "Adding a channel to a max fusion held to a fixed false-alarm "
              "budget raises the threshold, and a higher threshold costs "
              "detections that had nothing to do with the new channel. The "
              "third row separates the two explanations by scoring the "
              "three-channel monitor at the two-channel monitor's threshold.",
              "",
              "| dataset | condition | " +
              " | ".join(c.replace("det_", "")
                         for c in allocation.columns if c.startswith("det_")) +
              " | scoreable FA |",
              "|---|---|" + "---|" * (sum(c.startswith("det_")
                                          for c in allocation.columns) + 1)]
    detection_columns = [c for c in allocation.columns if c.startswith("det_")]
    for (corpus, condition), group in allocation.groupby(
            ["dataset", "condition"], sort=False):
        cells = " | ".join(f"{group[c].mean():.3f}" for c in detection_columns)
        lines.append(f"| {corpus} | {condition} | {cells} | "
                     f"{group['scoreable_fa_rate'].mean():.3f} |")
    ratios = allocation.groupby("dataset")["theta_ratio"].median()
    lines += ["",
              "Median theta(e+m+c) / theta(e+m): " +
              ", ".join(f"{d} {v:.2f}" for d, v in ratios.items()) + ".",
              "",
              "At the fold sizes these corpora allow, `pick_threshold` "
              "returns the single most extreme healthy episode: with 17 "
              "threshold episodes the order-statistic floor 1/(n+1) = 0.056 "
              "already exceeds the 5% budget, so no healthy episode sits "
              "above theta and one episode sets it. That is why a third "
              "channel moves theta at all. The protocol's own floor of 80 "
              "threshold episodes puts theta at the 4th largest of 80, and "
              "the untouched test size of 300 at the 15th of 300; neither "
              "exists yet.",
              ""]

    latency = arms.dropna(subset=["step_latency_p50_us"])
    lines += ["", "## Runtime cost", "",
              "Per-step scoring, measured in the same pass. An accuracy gain "
              "bought by making the monitor materially heavier is not free. "
              "Wall-clock is a property of the machine and of what else it "
              "was doing: read the RATIO between arms measured in one pass, "
              "not the absolute microseconds.",
              "", "| dataset | arm | p50 us/step | p95 us/step |",
              "|---|---|---|---|"]
    for _, r in latency.iterrows():
        lines.append(f"| {r['dataset']} | {r['arm']} | "
                     f"{r['step_latency_p50_us']:.1f} | "
                     f"{r['step_latency_p95_us']:.1f} |")

    lines += ["", "Scoring a third channel is a third ESN, so the per-step "
              "cost rises roughly in proportion -- the goal channel alone "
              "runs at about 52 us/step. Producing the channel and carrying "
              "it are separate costs, measured below: the four dims are set "
              "algebra over content tokens, and they widen the feature "
              "vector from 60 to 64.",
              "",
              "| dataset | steps | build us/step v4 -> v5 | bytes/step v4 -> "
              "v5 | dims |", "|---|---|---|---|---|"]
    for _, r in cost.iterrows():
        lines.append(
            f"| {r['dataset']} | {int(r['n_steps'])} | "
            f"{r['v4_grounding_build_us_per_step']:.0f} -> "
            f"{r['v5_goal_build_us_per_step']:.0f} | "
            f"{r['v4_grounding_feature_bytes_per_step']:.0f} -> "
            f"{r['v5_goal_feature_bytes_per_step']:.0f} | "
            f"{int(r['v4_grounding_dims'])} -> {int(r['v5_goal_dims'])} |")
    lines.append("")
    OUT_REPORT.write_text("\n".join(lines), "utf-8")


if __name__ == "__main__":
    main()
