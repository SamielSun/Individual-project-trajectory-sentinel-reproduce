"""Keep the pre-fix framework baseline independently executable.

The pre-fix result -- LangGraph detection 0.625 at 0.000 healthy false
alarms, AutoGen 0.913 at 0.125, and the zero-shot arm beside them -- is the
control every later E1 number is judged against. It has to stay runnable from
the tree, not merely protected by a recorded hash, and it must never be
satisfied by new detector logic wearing the old name.

Two things make that true and this tool checks both:

  * the freeze artifact reconstructs its monitor from ITS OWN recorded
    hyperparameters, so a later change to an `ESNEnsembleMonitor` default
    cannot reinterpret it (`load_frozen_monitor`);
  * every code file that can move either arm's numbers is pinned here by git
    blob hash, and a changed revision is refused unless it has been PROVEN to
    reconstruct the byte-identical monitor.

`--prove` is the only way to widen that allowlist, and it fails unless the
reconstruction digest matches. There is no flag to skip the check.

    py -m devtools.baseline_lock --write     # record the lock (once)
    py -m devtools.baseline_lock --check     # gate: still reproducible?
    py -m devtools.baseline_lock --prove     # accept a proven-equal revision
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = REPO_ROOT / "results" / "framework_monitor_freeze.json"
LOCK_PATH = REPO_ROOT / "results" / "framework_monitor_freeze_compat.json"

#: Everything that can change either published arm. The first five are the
#: freeze's own dependency chain; the rest are what the CALIBRATED arm adds,
#: which the freeze artifact never covered because it describes one monitor
#: rather than the study that used it.
DEPENDENCY_FILES = (
    "derail/monitor/esn.py",
    "derail/common.py",
    "derail/telemetry/adapter.py",
    "derail/telemetry/events.py",
    "derail/evaluation/metrics.py",
    "derail/monitor/deployment_calibration.py",
    "derail/experiments/framework_monitor_freeze.py",
    "derail/experiments/run_framework_real_tool_analysis.py",
    "derail/experiments/run_framework_generalized_monitor_eval.py",
)

#: The published artifacts this baseline consists of. Named so the lock says
#: what it is protecting, and so a reader can find the numbers.
PUBLISHED = (
    "results/framework_real_tool_report.md",
    "results/framework_generalized_monitor_report.md",
    "results/tables/framework_generalized_monitor_summary.csv",
    "results/tables/framework_generalized_monitor_alarms.csv",
    "results/tables/framework_real_tool_alarms.csv",
)

#: Corpora the digest scores. The zero-shot arm's own scoring targets, so the
#: digest covers the monitor as it was actually used, not a synthetic probe.
DIGEST_CORPORA = ("langgraph7b_real", "autogen7b_real")

#: The baseline is compared as NUMBERS, within a tolerance, not as a hash.
#:
#: These scores run through log and ridge solves, whose last place differs
#: between platform BLAS and libm builds, so a digest recorded on one machine
#: does not reproduce on another: the original rule rounded to six DECIMALS,
#: which on a CUSUM statistic reaching the high hundreds demanded about ten
#: significant digits of cross-platform agreement, and it failed the first
#: time the lock left the machine that wrote it.
#:
#: Rounding harder only moves the problem. Any hash of quantised floats has a
#: cliff at the rounding boundary, and its real margin is set by whichever
#: score happens to land nearest one -- measured at 3.5e-09 for six
#: significant figures, barely seven times the rule that already failed.
#: Comparing the scores themselves against a relative tolerance has no cliff:
#: the margin is the tolerance, at every magnitude, whatever the data does.
#:
#: The tolerance is far tighter than any real change in the detector. A
#: revision that alters what the monitor does moves these scores in their
#: first few digits, not their seventh.
DIGEST_RTOL = 1e-6
DIGEST_ATOL = 1e-9

MONITOR_VERSION = "esn_cusum_max[e,m]/v1-prefix"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=10).stdout.strip()


def _blob(path: str) -> str:
    """Blob hash of the STAGED content -- what the next commit would record.

    The index rather than HEAD, so a proof can be taken on exactly the change
    about to be committed. For an unmodified file the two are the same, so
    this only differs where it matters.
    """
    return _git("rev-parse", f":{path}")


def _dirty(paths: tuple[str, ...]) -> list[str]:
    """Dependency files whose working tree differs from what is staged.

    Blob hashes name staged content. An unstaged edit would therefore be
    invisible to the hash while still being what the interpreter imports, so
    a proof taken over it would attest to code that was never recorded.
    """
    # `git diff --name-only` is exactly "worktree differs from index". The
    # porcelain status columns are not parsed here on purpose: their leading
    # space is load-bearing, and any helper that strips output silently turns
    # an unstaged edit into a clean one.
    out = _git("diff", "--name-only", "--", *paths)
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def reference_scores(proving_equivalence: bool = False) -> dict:
    """The frozen monitor's own per-step scores, per episode.

    Stronger than any source hash: it is the monitor's behaviour on the
    episodes it was published against. Two trees agreeing here reconstruct the
    same detector whatever their source looks like -- and because the scores
    are kept rather than hashed, a tree that disagrees says which episode
    diverged and by how much.
    """
    from derail.experiments.framework_monitor_freeze import load_frozen_monitor
    from derail.telemetry.adapter import episode_from_trace

    monitor, _, _ = load_frozen_monitor(
        proving_equivalence=proving_equivalence)
    out = {}
    for corpus in DIGEST_CORPORA:
        corpus_dir = REPO_ROOT / "traces" / corpus
        manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
        for entry in sorted(manifest, key=lambda e: e["episode_id"]):
            steps = [json.loads(line) for line in
                     (corpus_dir / entry["file"]).read_text("utf-8").splitlines()
                     if line.strip()]
            ep = episode_from_trace(steps, entry["episode_id"],
                                    use_sentence_transformers=False,
                                    extended=True)
            monitor.start_episode()
            out[entry["episode_id"]] = [float(monitor.score_step(ep.X[t]))
                                        for t in range(ep.X.shape[0])]
    return out


def score_problems(locked: dict, current: dict) -> list[str]:
    """Where the reconstruction departs from the recorded baseline."""
    problems = []
    missing = sorted(set(locked) - set(current))
    extra = sorted(set(current) - set(locked))
    if missing:
        problems.append("%d recorded episodes are no longer scored, first %s"
                        % (len(missing), missing[0]))
    if extra:
        problems.append("%d episodes are scored that the baseline never "
                        "recorded, first %s" % (len(extra), extra[0]))
    for episode_id in sorted(set(locked) & set(current)):
        want = np.asarray(locked[episode_id], dtype=float)
        got = np.asarray(current[episode_id], dtype=float)
        if want.shape != got.shape:
            problems.append("%s: %d steps scored against %d recorded"
                            % (episode_id, got.size, want.size))
            continue
        if np.allclose(got, want, rtol=DIGEST_RTOL, atol=DIGEST_ATOL):
            continue
        off = np.abs(got - want) / np.maximum(np.abs(want), DIGEST_ATOL)
        worst = int(np.argmax(off))
        problems.append(
            "%s: step %d scores %.9g against the recorded %.9g "
            "(relative %.2e, tolerance %g)"
            % (episode_id, worst, got[worst], want[worst],
               off.max(), DIGEST_RTOL))
    return problems


def build_lock() -> dict:
    dirty = _dirty(DEPENDENCY_FILES)
    if dirty:
        raise SystemExit(
            f"[baseline-lock] REFUSED: {', '.join(dirty)} have unstaged "
            f"edits. A lock records staged blob hashes, so writing one now "
            f"would pin revisions that are not what this tree runs.")
    return {
        "purpose": "Keeps the pre-fix framework baseline executable and "
                   "unambiguous while the detector evolves. The freeze "
                   "artifact itself is immutable; this file records what has "
                   "been proven equivalent to it.",
        "monitor_version": MONITOR_VERSION,
        "freeze_artifact": FREEZE_PATH.relative_to(REPO_ROOT).as_posix(),
        "freeze_sha256": _sha256_file(FREEZE_PATH),
        "published_artifacts": list(PUBLISHED),
        "baseline_commit": _git("rev-parse", "HEAD"),
        "dependency_blob_hashes": {p: _blob(p) for p in DEPENDENCY_FILES},
        "compatible_dependency_blob_hashes": {},
        "reference_scores": reference_scores(),
        "digest_corpora": list(DIGEST_CORPORA),
        "digest_rtol": DIGEST_RTOL,
        "digest_atol": DIGEST_ATOL,
    }


def check(lock: dict) -> list[str]:
    """Everything wrong with the current tree, as a list of complaints."""
    problems = []
    if lock["freeze_sha256"] != _sha256_file(FREEZE_PATH):
        problems.append(
            f"{FREEZE_PATH.name} has changed since the lock was written; the "
            f"freeze is meant to be immutable")
    for path in _dirty(DEPENDENCY_FILES):
        problems.append(
            f"{path} has unstaged edits, so no blob hash describes what "
            f"this tree would actually run")
    allowed = lock.get("compatible_dependency_blob_hashes", {})
    for path, pinned in lock["dependency_blob_hashes"].items():
        current = _blob(path)
        if current == pinned or current in allowed.get(path, ()):
            continue
        problems.append(
            f"{path} is at {current or '(untracked)'}, which is neither the "
            f"baseline revision {pinned} nor a proven-equivalent one -- run "
            f"--prove")
    if problems:
        return problems                    # a digest now would just re-raise
    departures = score_problems(lock["reference_scores"],
                                reference_scores(proving_equivalence=True))
    if departures:
        problems.append("the frozen monitor no longer reproduces its own "
                        "scores; the pre-fix baseline is NOT reproducible "
                        "from this tree")
        problems.extend("  " + d for d in departures[:5])
    return problems


def prove(lock: dict) -> dict:
    """Record the current revisions as equivalent, if they demonstrably are."""
    dirty = _dirty(DEPENDENCY_FILES)
    if dirty:
        raise SystemExit(
            f"[baseline-lock] REFUSED: {', '.join(dirty)} have unstaged "
            f"edits; stage them before proving, or the proof records a blob "
            f"hash for code that was never run.")
    scores = reference_scores(proving_equivalence=True)
    departures = score_problems(lock["reference_scores"], scores)
    if departures:
        raise SystemExit(
            "[baseline-lock] REFUSED: this tree reconstructs a different "
            "monitor -- " + departures[0] + ". A revision that changes "
            "the frozen monitor's behaviour cannot be recorded as "
            "compatible with it -- give the new detector its own freeze.")
    allowed = lock.setdefault("compatible_dependency_blob_hashes", {})
    added = []
    for path, pinned in lock["dependency_blob_hashes"].items():
        current = _blob(path)
        if current and current != pinned and current not in allowed.get(path, ()):
            allowed.setdefault(path, []).append(current)
            added.append(f"{path} -> {current[:12]}")
    lock["proofs"] = lock.get("proofs", []) + [{
        "commit": _git("rev-parse", "HEAD"),
        "scores_matched": sum(len(v) for v in scores.values()),
        "added": added,
    }]
    return lock


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--prove", action="store_true")
    args = ap.parse_args(argv)

    if args.write:
        if LOCK_PATH.exists():
            raise SystemExit(
                f"[baseline-lock] {LOCK_PATH.name} already exists -- it "
                f"records a baseline that has already been established. Use "
                f"--prove to widen it, never --write to replace it.")
        LOCK_PATH.write_text(json.dumps(build_lock(), indent=2) + "\n", "utf-8")
        print(f"[baseline-lock] wrote {LOCK_PATH}")
        return

    lock = json.loads(LOCK_PATH.read_text("utf-8"))
    if args.prove:
        LOCK_PATH.write_text(json.dumps(prove(lock), indent=2) + "\n", "utf-8")
        print(f"[baseline-lock] proven equivalent; updated {LOCK_PATH}")
        return

    problems = check(lock)
    for problem in problems:
        print(f"[baseline-lock] {problem}")
    if problems:
        raise SystemExit(1)
    print(f"[baseline-lock] pre-fix baseline {lock['monitor_version']} "
          f"reproduces: "
          f"{sum(len(v) for v in lock['reference_scores'].values())} scores "
          f"within {lock['digest_rtol']:g} over "
          f"{len(lock['digest_corpora'])} corpora, "
          f"{len(lock['dependency_blob_hashes'])} dependencies pinned")


if __name__ == "__main__":
    main()
