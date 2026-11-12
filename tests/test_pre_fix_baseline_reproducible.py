"""Is the pre-fix framework baseline still executable from this tree?

Not "is it recorded", "is it hashed", or "did anyone promise not to touch
it" -- executable. These tests reconstruct the frozen monitor, re-run both
published arms against the committed corpora, and compare what comes out to
what was published: the zero-shot report and, from the calibrated arm, the
0.625 / 0.000 LangGraph and 0.913 / 0.125 AutoGen numbers.

The point is the control. Every later E1 result is read against this
baseline, so a version of it that quietly ran new detector logic would make
the comparison meaningless in exactly the way that is hardest to notice.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from devtools import baseline_lock
from derail.experiments import framework_monitor_freeze as freeze_module
from derail.experiments.framework_monitor_freeze import load_frozen_monitor

REPO_ROOT = baseline_lock.REPO_ROOT


def _lock() -> dict:
    return json.loads(baseline_lock.LOCK_PATH.read_text("utf-8"))


@pytest.mark.slow
def test_the_pre_fix_baseline_still_reconstructs_its_own_scores():
    """The lock's own gate: dependencies pinned or proven, digest reproduced."""
    assert baseline_lock.check(_lock()) == []


def test_the_frozen_monitor_is_built_from_its_artifact_not_from_defaults():
    monitor, theta, artifact = load_frozen_monitor()
    assert theta == artifact["theta_b5"]
    assert monitor.channels == tuple(artifact["channels"])
    assert len(monitor.subs) == len(artifact["channels"])
    outer = artifact["outer_monitor_config"]
    for sub in monitor.subs:
        assert sub.K == outer["K"]
        assert sub.cusum is outer["cusum"]
        for name, value in artifact["inner_esn_hyperparameters"].items():
            assert getattr(sub, name) == value, (
                f"{name} came from the current default rather than the "
                f"freeze artifact")


def test_the_frozen_split_is_the_one_the_artifact_records():
    _, _, artifact = load_frozen_monitor()
    train = artifact["training_episode_ids"]
    val = artifact["calibration_episode_ids"]
    test = artifact["held_out_test_episode_ids"]
    assert len(train) == artifact["training_episode_count"]
    assert len(val) == artifact["calibration_episode_count"]
    assert not (set(train) & set(val)), "train and calibration overlap"
    assert not (set(train) & set(test)), "train and held-out test overlap"
    assert not (set(val) & set(test)), "calibration and held-out test overlap"
    assert freeze_module._hash_ids(train) == artifact["training_input_hash"]
    assert freeze_module._hash_ids(val) == artifact["calibration_input_hash"]


def test_an_artifact_hyperparameter_wins_over_the_current_default(tmp_path):
    """Mutation check on the pinning itself.

    A copy of the freeze with one hyperparameter changed must produce a
    monitor carrying THAT value. If the reconstruction were still reading
    defaults, this would silently pass through the shipped 0.3 instead.
    """
    artifact = json.loads(freeze_module.OUT_PATH.read_text("utf-8"))
    assert artifact["inner_esn_hyperparameters"]["leak_rate"] != 0.7
    artifact["inner_esn_hyperparameters"]["leak_rate"] = 0.7
    path = tmp_path / "mutated_freeze.json"
    path.write_text(json.dumps(artifact), "utf-8")
    monitor, _, _ = load_frozen_monitor(path)
    assert all(sub.leak_rate == 0.7 for sub in monitor.subs)


def test_a_lock_recorded_against_a_different_freeze_is_refused(tmp_path,
                                                               monkeypatch):
    """No silent fallback: a stale allowlist must not wave anything through."""
    stale = tmp_path / "compat.json"
    stale.write_text(json.dumps({
        "freeze_sha256": "0" * 64,
        "compatible_dependency_blob_hashes": {"derail/common.py": ["cafe"]},
    }), "utf-8")
    monkeypatch.setattr(freeze_module, "COMPAT_PATH", stale)
    with pytest.raises(RuntimeError, match="recorded against a different"):
        freeze_module._proven_compatible_blobs(freeze_module.OUT_PATH)


def test_proving_compatibility_is_refused_when_behaviour_changed(monkeypatch):
    lock = _lock()
    moved = {k: [v + 1.0 for v in vs]
             for k, vs in lock["reference_scores"].items()}
    monkeypatch.setattr(baseline_lock, "reference_scores",
                        lambda proving_equivalence=False: moved)
    monkeypatch.setattr(baseline_lock, "_dirty", lambda paths: [])
    with pytest.raises(SystemExit, match="reconstructs a different monitor"):
        baseline_lock.prove(lock)


def test_proving_records_how_much_it_matched(monkeypatch):
    """The success path, which the refusal test above never reaches.

    `prove` walks the dependency files after comparing scores, and that loop
    binds its own name for each blob hash. A proof summary that read the
    wrong name would only fail here, once the loop has actually run.
    """
    lock = json.loads(json.dumps(_lock()))
    monkeypatch.setattr(baseline_lock, "_dirty", lambda paths: [])
    monkeypatch.setattr(baseline_lock, "reference_scores",
                        lambda proving_equivalence=False:
                        lock["reference_scores"])
    proven = baseline_lock.prove(lock)
    assert proven["proofs"][-1]["scores_matched"] == sum(
        len(v) for v in lock["reference_scores"].values())


def test_the_baseline_survives_last_place_arithmetic_but_not_a_real_change():
    """The reason the comparison is a tolerance and not a hash.

    These scores come out of log and ridge solves, so their last places
    differ between platform BLAS builds and a digest recorded on one machine
    does not reproduce on another. A relative nudge far larger than any such
    difference must still read as the same monitor, while a change of the
    size an altered detector produces must not.
    """
    locked = _lock()["reference_scores"]
    nudged = {k: [v * (1.0 + 1e-9) for v in vs] for k, vs in locked.items()}
    assert baseline_lock.score_problems(locked, nudged) == []

    changed = {k: [v * 1.01 for v in vs] for k, vs in locked.items()}
    problems = baseline_lock.score_problems(locked, changed)
    assert problems and "relative" in problems[0]


def test_a_dropped_episode_is_not_silently_ignored():
    locked = _lock()["reference_scores"]
    short = {k: v for k, v in list(locked.items())[1:]}
    assert any("no longer scored" in p
               for p in baseline_lock.score_problems(locked, short))


def test_the_first_changed_file_is_not_lost_when_parsing_git(monkeypatch):
    """Regression: the helper that runs git strips its output, which ate the
    leading status column of the FIRST line and turned an unstaged edit into
    a clean tree. The detection must not depend on leading whitespace."""
    monkeypatch.setattr(baseline_lock, "_git",
                        lambda *args: "derail/monitor/esn.py\n"
                                      "derail/common.py")
    assert baseline_lock._dirty(("derail/monitor/esn.py",
                                 "derail/common.py")) == [
        "derail/common.py", "derail/monitor/esn.py"]


def test_an_unstaged_dependency_edit_is_refused(monkeypatch):
    monkeypatch.setattr(baseline_lock, "_dirty",
                        lambda paths: ["derail/monitor/esn.py"])
    problems = baseline_lock.check(_lock())
    assert any("unstaged edits" in p for p in problems)


@pytest.mark.slow
def test_the_zero_shot_arm_reproduces_what_was_published(tmp_path, monkeypatch):
    from derail.experiments import run_framework_real_tool_analysis as study

    monkeypatch.setattr(study, "OUT_CSV", tmp_path / "alarms.csv")
    monkeypatch.setattr(study, "OUT_REPORT", tmp_path / "report.md")
    study.main()

    published = REPO_ROOT / "results" / "framework_real_tool_report.md"
    assert (tmp_path / "report.md").read_text("utf-8") == \
        published.read_text("utf-8")
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "alarms.csv"),
        pd.read_csv(REPO_ROOT / "results" / "tables" /
                    "framework_real_tool_alarms.csv"))


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:calibrating on")
def test_the_calibrated_arm_reproduces_the_published_rates(tmp_path,
                                                           monkeypatch):
    from derail.experiments import run_framework_generalized_monitor_eval \
        as study

    monkeypatch.setattr(study, "OUT_CSV", tmp_path / "alarms.csv")
    monkeypatch.setattr(study, "OUT_SUMMARY", tmp_path / "summary.csv")
    monkeypatch.setattr(study, "OUT_REPORT", tmp_path / "report.md")
    study.main()

    published = pd.read_csv(REPO_ROOT / "results" / "tables" /
                            "framework_generalized_monitor_summary.csv")
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "summary.csv"),
                                  published)
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "alarms.csv"),
        pd.read_csv(REPO_ROOT / "results" / "tables" /
                    "framework_generalized_monitor_alarms.csv"))

    # The two numbers the whole E1 comparison is read against, asserted by
    # value so a reader of this test can see what is being protected.
    calibrated = published[published["monitor"] == "calibrated"].set_index(
        "framework")
    assert calibrated.loc["langgraph", "detection_rate"] == pytest.approx(0.625)
    assert calibrated.loc["langgraph", "healthy_fa_rate"] == pytest.approx(0.0)
    assert calibrated.loc["autogen", "detection_rate"] == \
        pytest.approx(0.9130434782608695)
    assert calibrated.loc["autogen", "healthy_fa_rate"] == pytest.approx(0.125)


def test_a_derived_freeze_still_loads_once_a_dependency_has_drifted():
    """Regression: the lock proves CODE revisions, not one artifact.

    The compatibility check used to be taken against whatever freeze was
    being loaded rather than the one the lock describes. Once any dependency
    drifted, that made every derived artifact -- a mutation probe, an
    experiment -- fail with a stale-lock error instead of loading, because a
    derived file can never hash to the canonical one. The proof is about the
    dependency chain, so it must survive being applied to a different
    artifact.
    """
    import inspect

    assert freeze_module._proven_compatible_blobs() == \
        freeze_module._proven_compatible_blobs(freeze_module.OUT_PATH)
    caller = inspect.getsource(freeze_module.load_frozen_monitor)
    assert "_proven_compatible_blobs()" in caller, (
        "load_frozen_monitor must validate the lock against the freeze the "
        "lock describes, not against the artifact it was handed")
