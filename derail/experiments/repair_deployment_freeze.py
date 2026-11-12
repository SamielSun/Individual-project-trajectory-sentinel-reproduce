"""Fit and calibrate the repair path's monitor on ITS OWN deployment.

The repair study inherited the detector freeze's monitor, and that monitor was
fitted on `real_research7b` -- a different task collected by a different
harness. Measured, the rollback corpus's median step latency is 2.90s against
0.52s in the fit corpora, a 5.6x shift on a dimension that sits inside the
`m` channel. The consequence is not subtle: three of the four healthy episodes
in the rollback corpus alarm, so the monitor is reading a host difference as
derailment.

That rules out recalibrating a threshold and keeping the fit. A threshold can
absorb a shifted SCALE; it cannot absorb a shifted standardizer, because the
standardizer's own mean and variance for that dimension come from the fit
population and every downstream score is computed against them. So the monitor
is refitted here, on healthy runs from the collector the repair path actually
serves.

Fold sizes are the ones the committed sample-size study settled
(`results/calibration_sample_size.md`), not sizes chosen to fit the data
available: fit >= 140, threshold calibration >= 80, disjoint, plus an
untouched development fold that no fit, scale or threshold ever sees and every
healthy false-alarm number is measured on. Admission is the frozen rule -- the
task's own verifier must pass, and the episode must be long enough for the
detector to score at all.

The artifact records both arms of the transfer question, so the decision to
refit is evidence rather than assertion: the same folds are scored with a
monitor fitted here and with one fitted elsewhere.

Writes:
  results/repair_deployment_freeze.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from derail.common import rng_for
from derail.evaluation.metrics import first_alarm
from derail.experiments.repair_policy_freeze import OUT_PATH as POLICY_FREEZE
from derail.intervene.checkpoint import fit_grounding_reference
from derail.monitor.deployment_calibration import (_RECOMMENDED_CALIBRATION,
                                                   _RECOMMENDED_FIT,
                                                   calibrate_from_folds)
from derail.monitor.esn import _WASHOUT
from derail.telemetry.adapter import episode_from_trace

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_ROOT = REPO_ROOT / "traces"
OUT_PATH = REPO_ROOT / "results" / "repair_deployment_freeze.json"

#: Healthy runs from the SAME collector the repair path serves.
DEPLOYMENT_CORPUS = "real_task_rollback_healthy"
#: The corpus the repair study runs on. Its healthy episodes are the transfer
#: check, never a fold.
SERVED_CORPUS = "real_task_rollback"
#: What the detector freeze was fitted on. Scored here only to measure what
#: fitting on another deployment costs.
FOREIGN_CORPORA = ("langgraph7b_real2", "autogen7b_real2")

SPLIT_SEED = 5150
FA_BUDGET = 0.05
CHANNELS = ("e", "m")


def _load(corpus: str, *, grounding: bool):
    """Verified, scoreable healthy episodes under the frozen admission rule."""
    manifest = json.loads((TRACES_ROOT / corpus / "manifest.json")
                          .read_text("utf-8"))
    episodes, rejected = [], []
    for entry in sorted((e for e in manifest
                         if e.get("failure_class") is None),
                        key=lambda e: e["episode_id"]):
        steps = [json.loads(line) for line in
                 (TRACES_ROOT / corpus / entry["file"])
                 .read_text("utf-8").splitlines() if line.strip()]
        if entry.get("success") is not True:
            rejected.append({"episode_id": entry["episode_id"],
                             "reason": "task verifier did not pass"})
            continue
        if len(steps) <= _WASHOUT:
            rejected.append({"episode_id": entry["episode_id"],
                             "reason": f"T={len(steps)} <= washout {_WASHOUT}"})
            continue
        episodes.append(episode_from_trace(
            steps, f"{corpus}/{entry['episode_id']}",
            use_sentence_transformers=False,
            grounding=grounding, extended=not grounding))
    return episodes, rejected


def _split(episodes: list) -> tuple[list, list, list]:
    """fit / calibrate / untouched, disjoint, at the committed floors."""
    order = rng_for(SPLIT_SEED, "repair-deployment").permutation(len(episodes))
    shuffled = [episodes[i] for i in order]
    n_fit, n_cal = _RECOMMENDED_FIT, _RECOMMENDED_CALIBRATION
    if len(shuffled) < n_fit + n_cal:
        raise SystemExit(
            f"{len(shuffled)} verified healthy episodes in "
            f"{DEPLOYMENT_CORPUS}, but the committed protocol needs "
            f"{n_fit} fit + {n_cal} calibration (disjoint) before any "
            f"untouched fold. Collect more rather than shrinking the folds "
            f"-- see results/calibration_sample_size.md")
    return (shuffled[:n_fit], shuffled[n_fit:n_fit + n_cal],
            shuffled[n_fit + n_cal:])


def _alarm_rate(calibration, episodes: list) -> float:
    if not episodes:
        return float("nan")
    alarms = 0
    for ep in episodes:
        calibration.monitor.start_episode()
        stream = np.array([calibration.monitor.score_step(ep.X[t])
                           for t in range(ep.T)])
        alarms += first_alarm(stream, calibration.theta) is not None
    return alarms / len(episodes)


def _ids(episodes: list) -> list[str]:
    return [ep.episode_id for ep in episodes]


def build() -> dict:
    episodes, rejected = _load(DEPLOYMENT_CORPUS, grounding=False)
    fit, cal, untouched = _split(episodes)
    calibration = calibrate_from_folds(fit, cal, test_episodes=untouched,
                                       channels=CHANNELS,
                                       fa_budget=FA_BUDGET, seed=0)

    # Does fitting elsewhere actually cost anything? Same calibration and
    # untouched folds, so only the fit population differs.
    foreign = [ep for corpus in FOREIGN_CORPORA
               for ep in _load(corpus, grounding=False)[0]]
    foreign_fit = foreign[:_RECOMMENDED_FIT]
    transferred = calibrate_from_folds(foreign_fit, cal,
                                       test_episodes=untouched,
                                       channels=CHANNELS,
                                       fa_budget=FA_BUDGET, seed=0)

    served, _ = _load(SERVED_CORPUS, grounding=False)
    grounding_fit, _ = _load(DEPLOYMENT_CORPUS, grounding=True)
    fit_ids = set(_ids(fit))
    reference = fit_grounding_reference(
        [ep for ep in grounding_fit if ep.episode_id in fit_ids])

    return {
        "deployment_corpus": DEPLOYMENT_CORPUS,
        "served_corpus": SERVED_CORPUS,
        "channels": list(CHANNELS),
        "fa_budget": FA_BUDGET,
        "split_seed": SPLIT_SEED,
        "n_verified_healthy": len(episodes),
        "n_rejected": len(rejected),
        "rejections": rejected,
        "fit_episode_ids": _ids(fit),
        "calibration_episode_ids": _ids(cal),
        "untouched_episode_ids": _ids(untouched),
        "fit_input_hash": _hash(_ids(fit)),
        "calibration_input_hash": _hash(_ids(cal)),
        "untouched_input_hash": _hash(_ids(untouched)),
        "theta": calibration.theta,
        "grounding_reference": reference.as_record(),
        "transfer_evidence": {
            "note": ("Same calibration and untouched folds; only the fit "
                     "population differs. `fitted_here` is the deployment's "
                     "own healthy runs, `fitted_elsewhere` the corpora the "
                     "detector freeze used."),
            "fitted_here": {
                "theta": calibration.theta,
                "untouched_healthy_alarm_rate": _alarm_rate(calibration,
                                                            untouched),
                "served_healthy_alarm_rate": _alarm_rate(calibration, served),
            },
            "fitted_elsewhere": {
                "theta": transferred.theta,
                "untouched_healthy_alarm_rate": _alarm_rate(transferred,
                                                            untouched),
                "served_healthy_alarm_rate": _alarm_rate(transferred, served),
            },
        },
        "policy_freeze_sha256": hashlib.sha256(
            POLICY_FREEZE.read_bytes()).hexdigest()
        if POLICY_FREEZE.exists() else None,
    }


def _hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def main() -> None:
    artifact = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", "utf-8")
    evidence = artifact["transfer_evidence"]
    print(f"[repair-deployment] {artifact['n_verified_healthy']} verified "
          f"healthy ({artifact['n_rejected']} rejected); folds "
          f"{len(artifact['fit_episode_ids'])} fit / "
          f"{len(artifact['calibration_episode_ids'])} calibrate / "
          f"{len(artifact['untouched_episode_ids'])} untouched")
    for arm in ("fitted_here", "fitted_elsewhere"):
        row = evidence[arm]
        print(f"[repair-deployment] {arm:18s} theta={row['theta']:9.3f} "
              f"untouched healthy FA={row['untouched_healthy_alarm_rate']:.3f} "
              f"served healthy FA={row['served_healthy_alarm_rate']:.3f}")
    print(f"[repair-deployment] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()


def load_deployment_monitor():
    """Refit the frozen deployment monitor from its recorded folds.

    No model is persisted anywhere in this project -- fitting is deterministic
    from code, seed and data -- so loading means re-running the same fit
    against the same recorded episode ids, and asserting they still hash to
    what was frozen. If the corpus moved under it, this raises rather than
    scoring against a monitor nobody froze.

    Returns (monitor, theta, grounding_reference, artifact).
    """
    from derail.intervene.checkpoint import GroundingReference

    artifact = json.loads(OUT_PATH.read_text("utf-8"))
    episodes, _ = _load(artifact["deployment_corpus"], grounding=False)
    by_id = {ep.episode_id: ep for ep in episodes}

    folds = {}
    for name in ("fit", "calibration"):
        ids = artifact[f"{name}_episode_ids"]
        if _hash(ids) != artifact[f"{name}_input_hash"]:
            raise RuntimeError(f"frozen {name} id set has drifted")
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise RuntimeError(
                f"{len(missing)} frozen {name} episodes are no longer in "
                f"{artifact['deployment_corpus']}: {missing[:3]}")
        folds[name] = [by_id[i] for i in ids]

    calibration = calibrate_from_folds(
        folds["fit"], folds["calibration"],
        channels=tuple(artifact["channels"]),
        fa_budget=artifact["fa_budget"], seed=0)
    if abs(calibration.theta - artifact["theta"]) > 1e-9:
        raise RuntimeError(
            f"refit threshold {calibration.theta!r} does not match the "
            f"frozen {artifact['theta']!r}; the monitor is not the one that "
            f"was frozen")
    reference = GroundingReference.from_record(artifact["grounding_reference"])
    return calibration.monitor, calibration.theta, reference, artifact
