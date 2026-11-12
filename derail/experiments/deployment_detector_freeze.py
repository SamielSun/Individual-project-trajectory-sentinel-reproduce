"""Freeze the served detector on the deployment's own healthy population.

The pre-fix detector (`results/framework_monitor_freeze.json`) stays exactly
as it is. It is the immutable baseline every historical number rests on, and
nothing here writes to it or re-proves it. This module produces a SEPARATE,
versioned artifact for the detector the repair path actually serves.

Three things change relative to that baseline, each already measured:

  statistic     `rate` rather than the running CUSUM. The CUSUM grows with
                the number of scored steps, so its threshold is a function of
                trajectory length; false alarms by length band run
                0.00/0.05/0.25 for the CUSUM and flat for the rate.
  common scale  each channel is mapped through its own healthy null before
                the max. This is NOT a detection gain -- measured over paired
                splits it moves detection by less than a hundredth at Holm
                p = 1.000. What it buys is that the fused threshold stops
                being the semantic channel's own threshold, which is what it
                is when the raw channel magnitudes decide the max.
  fold layout   four disjoint healthy folds instead of two. The scales and
                the threshold cannot come from the episodes the readout was
                fitted on, and the reported false-alarm rate cannot come from
                the episodes the threshold was picked on.

The candidate goal-consistency channel is evaluated here rather than
deferred. It was previously undecidable, not rejected: on a 17-episode
threshold fold `pick_threshold` returns the single most extreme healthy
episode, because the order-statistic floor 1/(n+1) = 0.056 already exceeds
the 5% budget, so one episode set theta and the fused budget could not be
divided. On this population the threshold fold is 80 and theta is the 4th
largest, which is the size at which the question can actually be asked.

Both arms are scored on identical folds, so the only thing that differs
between them is which channels they fuse.

Usage:
  py -m derail.experiments.deployment_detector_freeze
  py -m derail.experiments.deployment_detector_freeze --report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from derail.common import Standardizer, rng_for
from derail.evaluation.metrics import first_alarm
from derail.monitor.esn import _WASHOUT
from derail.monitor.fusion_protocol import PROTOCOL
from derail.monitor.normalized_fusion import (NormalizedFusionMonitor,
                                              pick_fused_threshold, scoreable)
from derail.telemetry.adapter import episode_from_trace

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_ROOT = REPO_ROOT / "traces"
OUT_PATH = REPO_ROOT / "results" / "deployment_detector_freeze.json"
REPORT_PATH = REPO_ROOT / "results" / "deployment_detector_freeze.md"

#: Healthy runs from the collector the repair path serves.
DEPLOYMENT_CORPUS = "real_task_rollback_healthy"
#: The corpus the repair study runs on. Its injected episodes measure
#: detection; its four healthy episodes are a transfer check, never a fold.
SERVED_CORPUS = "real_task_rollback"

MONITOR_VERSION = "normalized_fusion[e+m]/rate/v2"
SPLIT_SEED = 8317
SEED = 0

#: Fold sizes, at the committed floors in results/calibration_sample_size.md.
#: The remainder is development validation and is never fitted, scaled or
#: thresholded on.
N_FIT = PROTOCOL.min_fit_episodes
N_SCALE = PROTOCOL.min_scale_episodes
N_THRESHOLD = PROTOCOL.min_threshold_episodes

#: The frozen protocol, and the candidate that adds the goal channel.
ARMS: tuple[tuple[str, ...], ...] = (("e", "m"), ("e", "m", "c"))
BASELINE_ARM = "e+m"


def arm_name(channels: tuple[str, ...]) -> str:
    return "+".join(channels)


# ------------------------------------------------------------------ loading

def _read(corpus: str, entry: dict) -> list[dict]:
    return [json.loads(line) for line
            in (TRACES_ROOT / corpus / entry["file"])
            .read_text("utf-8").splitlines() if line.strip()]


def load_deployment_healthy() -> tuple[list, list[dict]]:
    """Verified, scoreable healthy episodes under the frozen admission rule.

    Admission is the task's own predeclared verifier, decided at collection
    time and recorded in the manifest. Nothing here re-judges an episode:
    a run the verifier failed is not a healthy negative, and an episode the
    detector cannot score is not a quiet one.
    """
    manifest = json.loads((TRACES_ROOT / DEPLOYMENT_CORPUS / "manifest.json")
                          .read_text("utf-8"))
    episodes, refused = [], []
    for entry in sorted(manifest, key=lambda e: e["episode_id"]):
        if entry.get("failure_class") is not None:
            continue
        if entry.get("success") is not True:
            refused.append({"episode_id": entry["episode_id"],
                            "reason": "task verifier did not pass"})
            continue
        episode = episode_from_trace(
            _read(DEPLOYMENT_CORPUS, entry),
            f"{DEPLOYMENT_CORPUS}/{entry['episode_id']}",
            use_sentence_transformers=False, goal=True)
        if not scoreable(episode):
            refused.append({"episode_id": entry["episode_id"],
                            "reason": f"T={episode.T} <= washout {_WASHOUT}"})
            continue
        episodes.append(episode)
    return episodes, refused


def load_served() -> tuple[list, list]:
    """(injected episodes with ground truth, healthy transfer-check episodes)."""
    manifest = json.loads((TRACES_ROOT / SERVED_CORPUS / "manifest.json")
                          .read_text("utf-8"))
    injected, healthy = [], []
    for entry in sorted(manifest, key=lambda e: e["episode_id"]):
        episode = episode_from_trace(
            _read(SERVED_CORPUS, entry),
            f"{SERVED_CORPUS}/{entry['episode_id']}",
            tau=entry["tau"] if entry["failure_class"] else None,
            failure_class=entry["failure_class"],
            use_sentence_transformers=False, goal=True)
        (injected if entry["failure_class"] else healthy).append(episode)
    return injected, healthy


# ------------------------------------------------------------------ folding

def split(episodes: list) -> dict[str, list]:
    """Four disjoint healthy folds, shuffled once and shared by every arm."""
    order = rng_for(SPLIT_SEED, "deployment-detector-freeze").permutation(len(episodes))
    pool = [episodes[i] for i in order]
    need = N_FIT + N_SCALE + N_THRESHOLD
    if len(pool) < need:
        raise SystemExit(
            f"{len(pool)} verified, scoreable healthy episodes in "
            f"{DEPLOYMENT_CORPUS}, but the committed protocol needs "
            f"{N_FIT} fit + {N_SCALE} scale + {N_THRESHOLD} threshold "
            f"(disjoint) before any development fold. Collect more rather "
            f"than shrinking the folds -- see "
            f"results/calibration_sample_size.md")
    a, b = N_FIT, N_FIT + N_SCALE
    return {"fit": pool[:a], "scale": pool[a:b],
            "threshold": pool[b:b + N_THRESHOLD],
            "development": pool[b + N_THRESHOLD:]}


def build(folds: dict[str, list], channels: tuple[str, ...]):
    """Fit, scale and threshold an arm on folds that never overlap."""
    monitor = NormalizedFusionMonitor(
        Standardizer().fit(folds["fit"]), channels=channels,
        statistic=PROTOCOL.statistic, seed=SEED)
    monitor.fit(folds["fit"])
    monitor.fit_channel_scales(folds["scale"])
    theta, used, skipped = pick_fused_threshold(
        monitor, folds["threshold"], fa_budget=PROTOCOL.fa_budget)
    return monitor, theta, {"n_threshold_used": used,
                            "n_threshold_skipped": skipped}


# --------------------------------------------------------------- evaluation

def _wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval; the normal approximation is wrong at k=0."""
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.959963984540054, k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def evaluate(monitor, theta: float, folds: dict[str, list],
             injected: list, served_healthy: list) -> dict:
    """False alarms on episodes nothing was fitted or picked on."""
    def alarms(episodes: list) -> list[dict]:
        rows = []
        for ep in episodes:
            step = first_alarm(monitor.score_stream(ep), theta)
            rows.append({"episode_id": ep.episode_id, "T": int(ep.T),
                         "alarmed": step is not None,
                         "alarm_step": None if step is None else int(step),
                         "failure_class": ep.failure_class,
                         "tau": None if ep.tau is None else int(ep.tau)})
        return rows

    dev = alarms(folds["development"])
    transfer = alarms(served_healthy)
    inj = alarms(injected)

    fa_k = sum(r["alarmed"] for r in dev)
    fa_lo, fa_hi = _wilson(fa_k, len(dev))

    per_class: dict[str, dict] = {}
    for row in inj:
        cls = row["failure_class"]
        bucket = per_class.setdefault(cls, {"n": 0, "detected": 0,
                                            "pre_onset": 0, "lead": []})
        bucket["n"] += 1
        # Causal detection: an alarm at or after onset, never before it. An
        # episode whose FIRST alarm precedes its onset is a miss and is
        # counted as one -- the alarm cannot have been caused by a failure
        # that had not happened yet. It is also counted separately as a
        # pre-onset alarm, because a miss that fired early for an unrelated
        # reason has a different cause from one that never fired at all, and
        # collapsing the two hides which mechanism needs work.
        hit = row["alarmed"] and row["alarm_step"] >= row["tau"]
        bucket["detected"] += hit
        bucket["pre_onset"] += bool(row["alarmed"]
                                    and row["alarm_step"] < row["tau"])
        if hit:
            bucket["lead"].append(row["alarm_step"] - row["tau"])
    for cls, bucket in per_class.items():
        lo, hi = _wilson(bucket["detected"], bucket["n"])
        bucket["rate"] = bucket["detected"] / bucket["n"]
        bucket["ci95"] = [lo, hi]
        bucket["median_steps_after_onset"] = (
            float(np.median(bucket["lead"])) if bucket["lead"] else None)
        bucket.pop("lead")

    det_k = sum(b["detected"] for b in per_class.values())
    det_n = sum(b["n"] for b in per_class.values())
    pre_k = sum(b["pre_onset"] for b in per_class.values())

    # Detection stratified by post-onset runway -- the scoreable steps a
    # detector actually gets at or after onset. An episode that ends two
    # steps after it was hijacked and one that runs for ten are not the same
    # test, and pooling them reports the corpus's trajectory-length
    # distribution as though it were the detector's accuracy. Runway also
    # counts in framework-specific units: an orchestrator that emits its
    # reasoning as separate events yields more scoreable steps for the same
    # agent work, so cross-framework detection is only comparable within a
    # band.
    by_runway: dict[str, dict] = {}
    for row in inj:
        runway = max(0, row["T"] - max(row["tau"], _WASHOUT))
        band = "<=2" if runway <= 2 else ("3-5" if runway <= 5 else ">5")
        bucket = by_runway.setdefault(band, {"n": 0, "detected": 0})
        bucket["n"] += 1
        bucket["detected"] += bool(row["alarmed"]
                                   and row["alarm_step"] >= row["tau"])
    for band, bucket in by_runway.items():
        bucket["rate"] = bucket["detected"] / bucket["n"]
        lo, hi = _wilson(bucket["detected"], bucket["n"])
        bucket["ci95"] = [lo, hi]
    det_lo, det_hi = _wilson(det_k, det_n)
    tk = sum(r["alarmed"] for r in transfer)

    return {
        "theta": theta,
        "development_false_alarms": {
            "k": fa_k, "n": len(dev), "rate": fa_k / len(dev) if dev else None,
            "ci95": [fa_lo, fa_hi], "budget": PROTOCOL.fa_budget},
        "served_healthy_transfer_check": {
            "k": tk, "n": len(transfer),
            "rate": tk / len(transfer) if transfer else None},
        "detection": {"k": det_k, "n": det_n,
                      "rate": det_k / det_n if det_n else None,
                      "ci95": [det_lo, det_hi]},
        "pre_onset_alarms": {"k": pre_k, "n": det_n,
                             "rate": pre_k / det_n if det_n else None,
                             "note": ("Injected episodes whose first alarm "
                                      "precedes onset. Counted as misses in "
                                      "detection, reported here so the "
                                      "mechanism is visible.")},
        "detection_per_class": per_class,
        "detection_by_runway": by_runway,
        "runway_note": ("Misses at runway <= 2 are runway-limited: the episode "
                        "ended before enough post-onset evidence existed. They "
                        "are counted as misses, and reported separately so "
                        "they are not read as detector accuracy."),
        "per_episode": {"development": dev, "served_healthy": transfer,
                        "injected": inj},
    }


# ------------------------------------------------------------------ artifact

def _ids(episodes: list) -> list[str]:
    return [ep.episode_id for ep in episodes]


def _hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                        # noqa: BLE001
        return "unknown"


def channel_contribution(folds: dict[str, list], injected: list,
                         candidate: tuple[str, ...],
                         baseline: tuple[str, ...]) -> dict:
    """What the extra channel actually changes, step by step and episode by
    episode.

    A channel can be present, carry variance, and still decide nothing. The
    three quantities that separate those cases are: how often it wins the
    fusion, whether those wins land on the step that sets an episode's
    maximum, and whether any changed maximum crosses the threshold. Reporting
    only the last one would make an inert channel and a silenced channel look
    identical.
    """
    extra = candidate[len(baseline):]
    base_monitor, theta, _ = build(folds, baseline)
    cand_monitor, _, _ = build(folds, candidate)

    wins, scored = {c: 0 for c in candidate}, 0
    for ep in folds["threshold"] + folds["development"] + injected:
        cand_monitor.start_episode()
        for t in range(ep.T):
            cand_monitor.score_step(ep.X[t])
            values = cand_monitor._last_channel_values
            if values:
                wins[candidate[int(np.argmax(values))]] += 1
                scored += 1

    changed, largest, crossed = 0, 0.0, 0
    for ep in folds["threshold"] + folds["development"] + injected:
        a = float(base_monitor.score_stream(ep).max())
        b = float(cand_monitor.score_stream(ep).max())
        if not np.isclose(a, b):
            changed += 1
            largest = max(largest, abs(b - a))
            if (a >= theta) != (b >= theta):
                crossed += 1

    return {
        "added_channels": list(extra),
        "scored_steps": scored,
        "fusion_wins": wins,
        "win_share": {c: (wins[c] / scored if scored else None)
                      for c in candidate},
        "episodes_compared": len(folds["threshold"])
                             + len(folds["development"]) + len(injected),
        "episodes_whose_maximum_moved": changed,
        "largest_maximum_shift": largest,
        "episodes_that_crossed_the_threshold": crossed,
        "goal_drift_episodes_available": sum(
            1 for ep in injected if ep.failure_class == "goal_drift"),
    }


def split_sensitivity(episodes: list, injected: list, served_healthy: list,
                      replicates: int) -> dict:
    """How much of the frozen result is the fold draw rather than the detector.

    A freeze has to commit to one split, but a number read off one split is
    not an estimate of anything. This project has been bitten by that before:
    a single-split bootstrap manufactured significance that pooling across
    splits removed entirely. So the frozen split is reported alongside the
    spread of the same procedure over other draws, and the spread is what the
    uncertainty should be read from -- not the interval around the one draw.

    Selection plays no part: the frozen split is fixed in advance by
    SPLIT_SEED, and nothing here can change it.
    """
    fa_rates, det_rates, thetas = [], [], []
    for r in range(replicates):
        order = rng_for(SPLIT_SEED + 1 + r,
                        "deployment-detector-freeze-sensitivity"
                        ).permutation(len(episodes))
        pool = [episodes[i] for i in order]
        a, b = N_FIT, N_FIT + N_SCALE
        folds = {"fit": pool[:a], "scale": pool[a:b],
                 "threshold": pool[b:b + N_THRESHOLD],
                 "development": pool[b + N_THRESHOLD:]}
        monitor, theta, _ = build(folds, ("e", "m"))
        result = evaluate(monitor, theta, folds, injected, served_healthy)
        fa_rates.append(result["development_false_alarms"]["rate"])
        det_rates.append(result["detection"]["rate"])
        thetas.append(theta)

    def spread(values: list[float]) -> dict:
        arr = np.array(values, dtype=float)
        return {"min": float(arr.min()), "median": float(np.median(arr)),
                "max": float(arr.max()), "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}

    return {"replicates": replicates,
            "note": ("Alternative draws of the same procedure. The frozen "
                     "split is not among them and was not chosen from them."),
            "development_false_alarm_rate": spread(fa_rates),
            "detection_rate": spread(det_rates),
            "theta": spread(thetas)}


def build_artifact(sensitivity_replicates: int = 0) -> dict:
    episodes, refused = load_deployment_healthy()
    folds = split(episodes)
    injected, served_healthy = load_served()

    arms: dict[str, dict] = {}
    for channels in ARMS:
        monitor, theta, sizes = build(folds, channels)
        result = evaluate(monitor, theta, folds, injected, served_healthy)
        arms[arm_name(channels)] = {
            "channels": list(channels),
            "monitor_name": monitor.name,
            "channel_scales": [
                {"channel": c, "median": s.median, "dispersion": s.dispersion,
                 "n_steps": s.n_steps}
                for c, s in zip(channels, monitor.scales or [])],
            **sizes, **result}

    return {
        "purpose": ("The detector the repair path serves, frozen on the "
                    "deployment's own healthy population. Separate from and "
                    "additive to results/framework_monitor_freeze.json, which "
                    "is the immutable pre-fix baseline and is not touched."),
        "monitor_version": MONITOR_VERSION,
        "statistic": PROTOCOL.statistic,
        "normalize_channels": PROTOCOL.normalize_channels,
        "threshold_on": PROTOCOL.threshold_on,
        "fa_budget": PROTOCOL.fa_budget,
        "protocol": "four_fold_holdout",
        "baseline_freeze": "results/framework_monitor_freeze.json",
        "deployment_corpus": DEPLOYMENT_CORPUS,
        "served_corpus": SERVED_CORPUS,
        "admission_rule": PROTOCOL.healthy_admission,
        "split_seed": SPLIT_SEED,
        "monitor_seed": SEED,
        "adopted_arm": BASELINE_ARM,
        "fold_sizes": {k: len(v) for k, v in folds.items()},
        "fold_episode_ids": {k: _ids(v) for k, v in folds.items()},
        "fold_input_hashes": {k: _hash(_ids(v)) for k, v in folds.items()},
        "healthy_refused": refused,
        "served_injected_episode_ids": _ids(injected),
        "served_healthy_episode_ids": _ids(served_healthy),
        "arms": arms,
        "channel_contribution": channel_contribution(
            folds, injected, ("e", "m", "c"), ("e", "m")),
        "split_sensitivity": (
            split_sensitivity(episodes, injected, served_healthy,
                              sensitivity_replicates)
            if sensitivity_replicates else None),
        "code_commit": _commit(),
        "evidence": ["results/deployment_healthy_collection_audit.md",
                     "results/calibration_sample_size.md",
                     "results/fusion_protocol_decision.json",
                     "results/goal_channel_study.md"],
    }


def write() -> Path:
    OUT_PATH.write_text(json.dumps(build_artifact(), indent=2) + "\n", "utf-8")
    return OUT_PATH


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        prog="py -m derail.experiments.deployment_detector_freeze")
    ap.add_argument("--report", action="store_true",
                    help="print the comparison instead of only writing it")
    ap.add_argument("--sensitivity", type=int, default=0, metavar="N",
                    help="also measure the spread over N alternative fold "
                         "draws; the frozen split is unaffected")
    args = ap.parse_args()

    artifact = build_artifact(sensitivity_replicates=args.sensitivity)
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", "utf-8")
    print(f"[deployment-freeze] wrote {OUT_PATH}")
    print(f"[deployment-freeze] folds: "
          + ", ".join(f"{k} {v}" for k, v in artifact["fold_sizes"].items()))
    for name, arm in artifact["arms"].items():
        fa = arm["development_false_alarms"]
        det = arm["detection"]
        print(f"  {name:>7}: theta={arm['theta']:.4f}  "
              f"FA {fa['k']}/{fa['n']} = {fa['rate']:.3f} "
              f"[{fa['ci95'][0]:.3f},{fa['ci95'][1]:.3f}]  "
              f"detection {det['k']}/{det['n']} = {det['rate']:.3f}")
        if args.report:
            for cls, bucket in sorted(arm["detection_per_class"].items()):
                print(f"      {cls:>19}: {bucket['detected']}/{bucket['n']}")
    sens = artifact.get("split_sensitivity")
    if sens:
        fa, det = sens["development_false_alarm_rate"], sens["detection_rate"]
        print(f"  over {sens['replicates']} other fold draws: "
              f"FA {fa['min']:.3f}-{fa['max']:.3f} (median {fa['median']:.3f}), "
              f"detection {det['min']:.3f}-{det['max']:.3f} "
              f"(median {det['median']:.3f})")
