"""Forensic accounting of every miss and false alarm the per-deployment
calibrated monitor produced on the LangGraph/AutoGen real-tool corpora.

Diagnosis only. Nothing here fits, tunes or selects anything that a
published number depends on: it re-runs the exact calibration
`run_framework_generalized_monitor_eval.py` already ran (same class, same
seed, same 60/20/20 split) and then opens up the per-step interior of the
score stream that the alarm table only summarises.

The corpora it reads are spent as evaluation data by being read this way --
they are development data from here on, and the artifacts below are labelled
as diagnostic evidence, never as a performance claim.

Writes:
  results/tables/framework_miss_forensics_steps.csv   (one row per step)
  results/tables/framework_miss_rootcause.csv         (one row per episode)
  results/framework_miss_forensics.md                 (readable summary)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from derail.common import (ACTION_TYPES, CHANNEL_SLICES, D_SEM, IDX_COS_DRIFT,
                           IDX_CTX_RATIO, IDX_REASON_DEPTH, IDX_RETRY_COUNT,
                           IDX_SELF_CONSISTENCY, IDX_TASK_SIM,
                           IDX_TOOL_LATENCY, IDX_TOOL_SUCCESS)
from derail.evaluation.metrics import first_alarm
from derail.monitor.deployment_calibration import calibrate
from derail.monitor.esn import _WASHOUT
from derail.telemetry.adapter import episode_from_trace

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_ROOT = REPO_ROOT / "traces"
OUT_STEPS = REPO_ROOT / "results" / "tables" / "framework_miss_forensics_steps.csv"
OUT_EPISODES = REPO_ROOT / "results" / "tables" / "framework_miss_rootcause.csv"
OUT_REPORT = REPO_ROOT / "results" / "framework_miss_forensics.md"

FRAMEWORKS = {"langgraph7b_real2": "langgraph", "autogen7b_real2": "autogen"}
FA_BUDGET = 0.05
#: Channels the shipped monitor scores, and the diagnostic superset that adds
#: the already-computed derived channel x (goal/trajectory consistency,
#: retries, tool success). Both are label-free; x is added here only to ask
#: whether a missed failure is invisible to the TELEMETRY or merely to the
#: CHANNELS this monitor happens to read.
SHIPPED_CHANNELS = ("e", "m")
DIAGNOSTIC_CHANNELS = ("e", "m", "x")
#: A step counts as materially moved when its largest standardized feature
#: exceeds this quantile of the same statistic over the calibration fold --
#: a healthy-only reference, so it never reads a failure label.
MOVE_QUANTILE = 0.99


def feature_names(width: int) -> list[str]:
    """Human-readable name per feature dimension, for `moved_dims`."""
    names = [f"e{i}" for i in range(D_SEM)]
    names += ["unc_mean", "unc_max", "unc_slope", "unc_high_frac"]
    names += [f"action_{a}" for a in ACTION_TYPES]
    names += ["latency_log", "outlen_log", "error_flag"]
    ext = {IDX_COS_DRIFT: "cos_drift", IDX_TASK_SIM: "task_sim",
           IDX_TOOL_SUCCESS: "tool_success", IDX_RETRY_COUNT: "retry_count",
           IDX_TOOL_LATENCY: "tool_latency", IDX_CTX_RATIO: "ctx_ratio",
           IDX_REASON_DEPTH: "reason_depth",
           IDX_SELF_CONSISTENCY: "self_consistency"}
    names += [ext[i] for i in sorted(ext)]
    return names[:width]


def load_corpus(corpus: str) -> tuple[list, dict]:
    """Episodes plus their manifest entries, exactly as the evaluation reads
    them (extended telemetry, no sentence-transformers)."""
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
            use_sentence_transformers=False, extended=True))
        meta[entry["episode_id"]] = entry
    return episodes, meta


def channel_trace(sub, episode) -> dict[str, np.ndarray]:
    """Per-step interior of one channel's ESN-CUSUM stream.

    Returns the CUSUM state actually emitted, plus the pre-CUSUM quantities
    it is built from: the raw fused increment and its two z-scored parts.
    Those are recovered exactly from the monitor's own EWMA recurrence
    (`ewma_t = a*raw_t + (1-a)*ewma_{t-1}`), so nothing is re-derived from a
    reimplementation of the scoring maths.
    """
    sub.start_episode()
    U = sub.standardizer.transform(episode.X)[:, sub._cols]
    a = sub.ewma_alpha
    cusum, raw, z_sup, z_dis, scored = [], [], [], [], []
    prev = (0.0, 0.0, 0.0)
    for t in range(episode.T):
        sub._advance(U[t])
        now = sub._ewma
        if t < _WASHOUT:
            raw.append(0.0); z_sup.append(0.0); z_dis.append(0.0)
            scored.append(False)
        else:
            raw.append((now[0] - (1.0 - a) * prev[0]) / a)
            z_sup.append((now[1] - (1.0 - a) * prev[1]) / a)
            z_dis.append((now[2] - (1.0 - a) * prev[2]) / a)
            scored.append(True)
        cusum.append(sub._cusum_state[0])
        prev = now
    return {"cusum": np.array(cusum), "raw": np.array(raw),
            "z_surprise": np.array(z_sup), "z_disagreement": np.array(z_dis),
            "scored": np.array(scored)}


def episode_streams(monitor, episode) -> dict:
    """Per-channel traces plus the max-fused stream the monitor emits."""
    per_channel = {sub.channels[0]: channel_trace(sub, episode)
                   for sub in monitor.subs}
    fused = np.max([c["cusum"] for c in per_channel.values()], axis=0)
    return {"per_channel": per_channel, "fused": fused}


def per_dim_reference(standardizer, healthy: list, cols: np.ndarray) -> np.ndarray:
    """Per-dimension healthy ceiling: the q0.99 of |z| for each feature over
    the calibration fold. Per-dimension rather than a single global ceiling,
    because a narrow channel moving hard is exactly the case a global max|z|
    hides."""
    stacked = np.concatenate(
        [np.abs(standardizer.transform(ep.X)[:, cols]) for ep in healthy])
    return np.quantile(stacked, MOVE_QUANTILE, axis=0)


def moved_dims(standardizer, episode, cols, ref: np.ndarray,
               start: int) -> tuple[list, float]:
    """Feature dimensions whose standardized value exceeds their own healthy
    ceiling anywhere from `start` on, strongest first (as a multiple of that
    ceiling), plus the largest such multiple."""
    names = feature_names(episode.X.shape[1])
    window = np.abs(standardizer.transform(episode.X)[start:, :][:, cols])
    if window.size == 0:
        return [], 0.0
    peak = window.max(axis=0)
    ratio = peak / np.where(ref > 0, ref, np.inf)
    order = np.argsort(ratio)[::-1]
    hits = [(names[cols[i]], round(float(peak[i]), 2), round(float(ratio[i]), 2))
            for i in order if ratio[i] > 1.0][:6]
    return hits, float(ratio.max())


def first_observable(standardizer, episode, cols, ref: np.ndarray,
                     tau: int | None) -> float:
    """First step at or after onset where any feature clears its own healthy
    ceiling -- the earliest step at which the trace itself differs from
    healthy behaviour, independent of any monitor."""
    Z = np.abs(standardizer.transform(episode.X)[:, cols])
    for t in range(0 if tau is None else int(tau), episode.T):
        if np.any(Z[t] > ref):
            return float(t)
    return float("nan")


def episode_statistics(cusum: np.ndarray, raw: np.ndarray) -> dict[str, float]:
    """Causal episode-level summaries of one score stream.

    Every one is computable online at the step it peaks:
      max_cusum      what the monitor ships -- the running CUSUM's high-water
                     mark, which grows with the number of scored steps;
      cusum_rate     the same high-water mark per scored step, so a short
                     episode is judged on the same scale as a long one;
      max_increment  the largest single-step evidence, with no accumulation
                     at all.
    """
    scored = np.arange(len(cusum)) - _WASHOUT + 1
    rate = np.where(scored > 0, cusum / np.maximum(scored, 1), 0.0)
    return {"max_cusum": float(cusum.max()),
            "cusum_rate": float(rate.max()),
            "max_increment": float(raw.max()) if raw.size else 0.0}




#: Label-free counterfactual arms. Each is calibrated on the SAME healthy
#: fold at the SAME budget as the shipped monitor, so "would this have
#: alarmed" is asked at equal false-alarm cost, never by lowering the bar.
COUNTERFACTUAL_ARMS = ("e", "m", "x")
STATISTIC_NAMES = ("max_cusum", "cusum_rate", "max_increment")


def _fused_traces(monitor, episode) -> tuple[np.ndarray, np.ndarray, dict]:
    """(fused CUSUM stream, fused raw-increment stream, per-channel traces)."""
    per_channel = {sub.channels[0]: channel_trace(sub, episode)
                   for sub in monitor.subs}
    fused = np.max([c["cusum"] for c in per_channel.values()], axis=0)
    fused_raw = np.max([c["raw"] for c in per_channel.values()], axis=0)
    return fused, fused_raw, per_channel


def _arm_thresholds(monitor, val: list) -> dict[str, float]:
    """One threshold per episode-level statistic, picked on the calibration
    fold at `FA_BUDGET` by the same empirical rule the monitor ships."""
    per_stat: dict[str, list[float]] = {n: [] for n in STATISTIC_NAMES}
    for ep in val:
        cusum, raw, _ = _fused_traces(monitor, ep)
        for name, value in episode_statistics(cusum, raw).items():
            per_stat[name].append(value)
    return {name: float(np.quantile(values, 1.0 - FA_BUDGET, method="higher"))
            for name, values in per_stat.items()}


def classify(row: dict) -> str:
    """Root cause of one miss or false alarm, from the measured interior.

    Ordered so the first mechanism that is sufficient on its own wins: a
    trajectory with no scored step cannot be diagnosed as a threshold
    problem, and a stream that never leaves zero cannot be diagnosed as slow
    accumulation.
    """
    if row["outcome"] == "false_alarm":
        if row["alt_arms_silent"]:
            return "decision_statistic_admits_healthy_transient"
        return "healthy_transient_resembles_failure"
    if row["n_scored_steps"] == 0:
        return "trajectory_too_short_to_score"
    if row["n_post_onset_scored_steps"] == 0:
        return "washout_suppresses_post_onset_evidence"
    if row["episode_max_fused"] == 0.0:
        # A stream pinned at zero has two very different causes: evidence
        # worth less than the CUSUM's slack, or evidence pointing the wrong
        # way entirely. Measured at k=0 the slack turns out not to be
        # binding on this corpus, so the two are separated here rather than
        # both being attributed to the slack.
        return ("residual_below_healthy_centre"
                if row["max_post_onset_increment"] <= 0.0
                else "cusum_slack_clips_evidence")
    if row["alt_arms_alarming"]:
        same_channels = [a for a in row["alt_arms_alarming"].split(";")
                         if a.startswith("em:")]
        return ("decision_statistic_loses_signal" if same_channels
                else "informative_channel_lost_in_fusion")
    if not row["evidence_before_termination"]:
        return "telemetry_lacks_signal"
    return "cusum_accumulates_too_slowly"


def analyse(corpus: str, framework: str) -> tuple[list[dict], list[dict], dict]:
    episodes, meta = load_corpus(corpus)
    healthy = [ep for ep in episodes if ep.is_healthy]
    injected = [ep for ep in episodes if not ep.is_healthy]
    width = episodes[0].X.shape[1]

    cal = calibrate(healthy, channels=SHIPPED_CHANNELS, fa_budget=FA_BUDGET,
                    seed=0)
    val = [ep for ep in healthy if ep.episode_id in cal.val_ids]
    test = [ep for ep in healthy if ep.episode_id in cal.test_ids]
    scored_set = test + injected

    # "Did anything move" must be asked of the whole telemetry vector, not
    # only of the channels this monitor happens to read -- otherwise a signal
    # sitting in the derived channel x is reported as absent. u is excluded
    # because these frameworks surface no logprobs, so it is identically 0.
    evidence_cols = np.concatenate([np.arange(width)[CHANNEL_SLICES[c]]
                                    for c in ("e", "m", "x")])
    ref = per_dim_reference(cal.standardizer, val, evidence_cols)

    # Every counterfactual arm shares this calibration fold and this budget.
    arms = {"em": cal}
    arms.update({c: calibrate(healthy, channels=(c,), fa_budget=FA_BUDGET,
                              seed=0) for c in COUNTERFACTUAL_ARMS})
    thresholds = {name: _arm_thresholds(arm.monitor, val)
                  for name, arm in arms.items()}

    step_rows, ep_rows = [], []
    for ep in scored_set:
        fused, fused_raw, per_channel = _fused_traces(cal.monitor, ep)
        alarm = first_alarm(fused, cal.theta)
        entry = meta[ep.episode_id]
        tau = None if ep.is_healthy else int(ep.tau)
        onset = 0 if tau is None else tau
        n_scored = max(0, ep.T - _WASHOUT)
        n_post = max(0, ep.T - max(onset, _WASHOUT))
        outcome = ("false_alarm" if ep.is_healthy and alarm is not None else
                   "correct_silence" if ep.is_healthy else
                   "miss" if alarm is None else
                   "early_alarm" if alarm < tau else "true_alarm")

        alarming, silent, stat_values = [], [], {}
        for arm_name, arm in arms.items():
            a_cusum, a_raw, _ = _fused_traces(arm.monitor, ep)
            for stat, value in episode_statistics(a_cusum, a_raw).items():
                key = f"{arm_name}:{stat}"
                stat_values[key] = round(value, 3)
                if key == "em:max_cusum":
                    continue                      # the shipped rule itself
                (alarming if value > thresholds[arm_name][stat]
                 else silent).append(key)

        hits, ratio = moved_dims(cal.standardizer, ep, evidence_cols, ref,
                                 onset)
        for t in range(ep.T):
            row = {"framework": framework, "dataset": corpus,
                   "episode_id": ep.episode_id,
                   "t": t, "is_washout": t < _WASHOUT,
                   "is_post_onset": tau is not None and t >= tau,
                   "fused_cusum": fused[t], "fused_raw_increment": fused_raw[t],
                   "theta": cal.theta, "margin_to_theta": fused[t] - cal.theta}
            for name, tr in per_channel.items():
                row[f"{name}_cusum"] = tr["cusum"][t]
                row[f"{name}_raw_increment"] = tr["raw"][t]
                row[f"{name}_z_surprise"] = tr["z_surprise"][t]
                row[f"{name}_z_disagreement"] = tr["z_disagreement"][t]
            step_rows.append(row)

        ep_row = {
            "framework": framework, "dataset": corpus,
            "episode_id": ep.episode_id, "task": entry.get("task"),
            "failure_class": ep.failure_class, "is_healthy": ep.is_healthy,
            "T": ep.T, "tau": tau, "outcome": outcome,
            "detected": outcome == "true_alarm",
            "first_observable_step": first_observable(
                cal.standardizer, ep, evidence_cols, ref, tau),
            "first_scored_step": _WASHOUT if ep.T > _WASHOUT else None,
            "alarm_step": float("nan") if alarm is None else float(alarm),
            "n_scored_steps": n_scored,
            "n_post_onset_scored_steps": n_post,
            "washout_cost_steps": max(0, min(_WASHOUT, ep.T) - onset),
            "theta": cal.theta,
            "episode_max_fused": float(fused.max()),
            "max_post_onset_increment": float(
                fused_raw[max(onset, _WASHOUT):].max())
                if ep.T > max(onset, _WASHOUT) else float("nan"),
            "margin_to_theta": float(fused.max() - cal.theta),
            "peak_dim_ratio_to_healthy": round(ratio, 2),
            "moved_dims": ";".join(f"{n}={v}(x{r})" for n, v, r in hits),
            "evidence_before_termination": bool(hits),
            "alt_arms_alarming": ";".join(sorted(alarming)),
            "alt_arms_silent": ";".join(sorted(silent)),
        }
        ep_row.update({f"stat_{k.replace(':', '_')}": v
                       for k, v in stat_values.items()})
        winner = np.array([max(per_channel, key=lambda c: per_channel[c]["cusum"][t])
                           for t in range(ep.T)])
        ep_row["fusion_winner_by_step"] = "".join(winner)
        ep_row["fusion_won_by_e_frac"] = round(
            float(np.mean(winner == "e")), 3)
        for name, tr in per_channel.items():
            ep_row[f"{name}_cusum_by_step"] = json.dumps(
                [round(float(v), 3) for v in tr["cusum"]])
            ep_row[f"{name}_raw_increment_by_step"] = json.dumps(
                [round(float(v), 3) for v in tr["raw"]])
        ep_row["fused_cusum_by_step"] = json.dumps(
            [round(float(v), 3) for v in fused])
        ep_row["root_cause"] = (
            classify(ep_row) if outcome in ("miss", "false_alarm", "early_alarm")
            else "")
        ep_rows.append(ep_row)

    val_maxima = []
    for ep in val:
        c, _, _ = _fused_traces(cal.monitor, ep)
        val_maxima.append(float(c.max()))
    context = {
        "framework": framework, "theta": cal.theta,
        "n_val": len(val), "n_test": len(test), "n_injected": len(injected),
        "val_maxima": sorted(val_maxima),
        "n_val_zero": sum(1 for v in val_maxima if v == 0.0),
        "val_T": sorted(ep.T for ep in val),
        "thresholds": thresholds,
    }
    return step_rows, ep_rows, context


def _report(contexts: list[dict], episodes: pd.DataFrame) -> str:
    lines = ["# Forensic accounting: every miss and false alarm",
             "",
             "**SUPPORTING / diagnostic evidence, not a performance claim.** "
             "This is an interior view of a result that has already been "
             "reported (`results/framework_generalized_monitor_report.md`), "
             "which stands unchanged as the pre-fix baseline. Reading these "
             "corpora this way spends them: they are development data from "
             "here on.",
             "",
             "Every counterfactual arm below is calibrated on the SAME "
             "healthy fold at the SAME 5% budget as the shipped monitor, so "
             "'would this have alarmed' is always asked at equal false-alarm "
             "cost. `cusum_rate` divides the running CUSUM by the number of "
             "scored steps, `max_increment` drops accumulation entirely; both "
             "are causal and both peak at a step the monitor could act on.",
             ""]
    for ctx in contexts:
        maxima = np.array(ctx["val_maxima"])
        lines += [f"## {ctx['framework']}", "",
                  f"- calibration fold: {ctx['n_val']} healthy episodes, "
                  f"T = {min(ctx['val_T'])}..{max(ctx['val_T'])}; held-out "
                  f"healthy test {ctx['n_test']}, injected {ctx['n_injected']}",
                  f"- shipped `theta` (max_cusum, empirical, "
                  f"{FA_BUDGET:.0%}): **{ctx['theta']:.3f}**",
                  f"- calibration-fold episode maxima: min {maxima.min():.2f}, "
                  f"median {np.median(maxima):.2f}, max {maxima.max():.2f}; "
                  f"{ctx['n_val_zero']} of {ctx['n_val']} are exactly 0.0",
                  "- budget-matched thresholds per arm and statistic:", ""]
        lines.append("  | arm | " + " | ".join(STATISTIC_NAMES) + " |")
        lines.append("  |---|" + "---|" * len(STATISTIC_NAMES))
        for arm, stats in ctx["thresholds"].items():
            lines.append(f"  | {arm} | " + " | ".join(
                f"{stats[s]:.3f}" for s in STATISTIC_NAMES) + " |")
        lines.append("")

    failures = episodes[episodes["root_cause"] != ""]
    lines += ["## Every miss and false alarm", "",
              "| framework | episode | class | T | tau | outcome | max fused | "
              "theta | moved dims (post-onset peak, x healthy ceiling) | "
              "budget-matched arms that would alarm | root cause |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in failures.iterrows():
        tau = "" if r["tau"] is None or pd.isna(r["tau"]) else int(r["tau"])
        lines.append(
            f"| {r['framework']} | {r['episode_id']} | "
            f"{r['failure_class'] if isinstance(r['failure_class'], str) else 'healthy'} | "
            f"{r['T']} | {tau} | {r['outcome']} | "
            f"{r['episode_max_fused']:.2f} | {r['theta']:.2f} | "
            f"{r['moved_dims'] or '(none above ceiling)'} | "
            f"{r['alt_arms_alarming'] or '(none)'} | {r['root_cause']} |")
    lines += ["", "## Root-cause counts", ""]
    for cause, n in failures["root_cause"].value_counts().items():
        lines.append(f"- **{cause}**: {n}")
    lines += ["", "## Per-class detection on this development corpus", ""]
    inj = episodes[~episodes["is_healthy"]]
    lines += ["| framework | class | detected / n |", "|---|---|---|"]
    for (fw, cls), grp in inj.groupby(["framework", "failure_class"]):
        lines.append(f"| {fw} | {cls} | {int(grp['detected'].sum())} / "
                     f"{len(grp)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    step_rows, ep_rows, contexts = [], [], []
    for corpus, framework in FRAMEWORKS.items():
        s, e, ctx = analyse(corpus, framework)
        step_rows += s
        ep_rows += e
        contexts.append(ctx)
        print(f"[miss-forensics] {framework}: {len(e)} episodes scored, "
              f"{sum(1 for r in e if r['root_cause'])} to explain")

    steps = pd.DataFrame(step_rows)
    episodes = pd.DataFrame(ep_rows)
    OUT_STEPS.parent.mkdir(parents=True, exist_ok=True)
    steps.to_csv(OUT_STEPS, index=False)
    episodes.to_csv(OUT_EPISODES, index=False)
    OUT_REPORT.write_text(_report(contexts, episodes), "utf-8")
    for path in (OUT_STEPS, OUT_EPISODES, OUT_REPORT):
        print(f"[miss-forensics] wrote {path}")
    print(episodes[episodes["root_cause"] != ""]["root_cause"]
          .value_counts().to_string())


if __name__ == "__main__":
    main()
