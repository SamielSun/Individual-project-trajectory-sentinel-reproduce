"""The repair path may not calibrate itself on a smaller sample than the
committed protocol allows.

Item 6 settled the fold sizes on measured evidence (fit >= 140, threshold
calibration >= 80, disjoint). The temptation when a study is nearly ready is
to shrink them to whatever has been collected, which converts a measured
standard into whatever was convenient. These tests make that fail loudly.
"""
from __future__ import annotations

import inspect
import json

import pytest

from derail.experiments import repair_deployment_freeze as freeze
from derail.monitor.deployment_calibration import (_RECOMMENDED_CALIBRATION,
                                                   _RECOMMENDED_FIT)


class _Ep:
    def __init__(self, i: int) -> None:
        self.episode_id = f"ep{i}"
        self.T = 8


def test_the_split_refuses_to_shrink_below_the_committed_floors():
    too_few = [_Ep(i) for i in range(_RECOMMENDED_FIT + _RECOMMENDED_CALIBRATION - 1)]
    with pytest.raises(SystemExit, match="calibration_sample_size"):
        freeze._split(too_few)


def test_the_split_is_disjoint_and_meets_the_floors():
    pool = [_Ep(i) for i in range(_RECOMMENDED_FIT + _RECOMMENDED_CALIBRATION + 40)]
    fit, cal, untouched = freeze._split(pool)
    assert len(fit) >= _RECOMMENDED_FIT
    assert len(cal) >= _RECOMMENDED_CALIBRATION
    ids = [{e.episode_id for e in fold} for fold in (fit, cal, untouched)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(f) for f in (fit, cal, untouched)) == len(pool)


def test_the_untouched_fold_is_never_fitted_or_thresholded():
    """Structural: `build` passes the untouched fold only as `test_episodes`,
    which `calibrate_from_folds` records and does not learn from."""
    source = inspect.getsource(freeze.build)
    assert "test_episodes=untouched" in source
    assert "fit_episodes=untouched" not in source
    assert "calibrate_from_folds(untouched" not in source


def test_admission_is_the_frozen_rule():
    """Verified AND scoreable, with rejections counted rather than dropped."""
    source = inspect.getsource(freeze._load)
    assert 'entry.get("success") is not True' in source, (
        "healthy admission must require the task verifier to pass")
    assert "_WASHOUT" in source, (
        "an episode the detector cannot score is not a healthy negative")
    assert "rejected.append" in source, "rejections must be counted, not dropped"


def test_the_transfer_question_is_answered_with_both_arms():
    """The decision to refit has to rest on a measurement, so the artifact
    must carry the comparison, not just the winner."""
    source = inspect.getsource(freeze.build)
    assert "fitted_here" in source and "fitted_elsewhere" in source
    assert "foreign_fit" in source


def test_the_loader_refuses_a_monitor_it_did_not_freeze():
    source = inspect.getsource(freeze.load_deployment_monitor)
    assert "id set has drifted" in source
    assert "does not match the" in source, (
        "a refit whose threshold differs must raise, not be used")


@pytest.mark.skipif(not freeze.OUT_PATH.exists(),
                    reason="deployment freeze not built yet")
def test_the_committed_freeze_respects_its_own_protocol():
    artifact = json.loads(freeze.OUT_PATH.read_text("utf-8"))
    assert len(artifact["fit_episode_ids"]) >= _RECOMMENDED_FIT
    assert len(artifact["calibration_episode_ids"]) >= _RECOMMENDED_CALIBRATION
    folds = [set(artifact[k]) for k in ("fit_episode_ids",
                                        "calibration_episode_ids",
                                        "untouched_episode_ids")]
    assert not (folds[0] & folds[1])
    assert not (folds[0] & folds[2])
    assert not (folds[1] & folds[2])
    assert artifact["untouched_episode_ids"], (
        "no untouched fold, so healthy false alarms have nowhere to be "
        "measured that the threshold did not already see")
