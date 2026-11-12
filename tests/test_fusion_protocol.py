"""The protocol decision must not drift from the code or from its evidence.

A decision record is only worth something if it stays true. These pin it to
the module that defines it, to the sample-size floors it claims to respect,
and to the committed studies it cites.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from derail.monitor.deployment_calibration import (_RECOMMENDED_CALIBRATION,
                                                   _RECOMMENDED_FIT)
from derail.monitor.fusion_protocol import PROTOCOL, RECORD_PATH, REPO_ROOT
from derail.monitor.normalized_fusion import STATISTICS


def test_the_committed_record_matches_the_module():
    assert RECORD_PATH.exists(), "the protocol decision was never written out"
    assert json.loads(RECORD_PATH.read_text("utf-8")) == PROTOCOL.as_record()


def test_the_protocol_is_buildable():
    assert PROTOCOL.statistic in STATISTICS
    assert PROTOCOL.channels == ("e", "m")
    assert PROTOCOL.threshold_on == "fused_statistic"
    assert 0.0 < PROTOCOL.fa_budget < 1.0


def test_the_fold_sizes_respect_the_measured_floors():
    """The learning curve is the authority on these, not this file."""
    assert PROTOCOL.min_fit_episodes >= _RECOMMENDED_FIT
    assert PROTOCOL.min_threshold_episodes >= _RECOMMENDED_CALIBRATION


def test_the_untouched_test_size_can_carry_the_false_alarm_claim():
    """At the chosen N, a realistic 2% observed rate must still bound below
    the 5% budget -- otherwise the evaluation cannot say what it is for."""
    precision = pd.read_csv(REPO_ROOT / "results" / "tables" /
                            "calibration_fa_precision.csv")
    row = precision[(precision["n_untouched_healthy"] ==
                     PROTOCOL.untouched_test_episodes) &
                    (precision["observed_fa"].round(3) == 0.020)]
    assert not row.empty, (
        f"no precision row for N={PROTOCOL.untouched_test_episodes} at 2% "
        f"observed; the chosen size is not backed by the committed table")
    assert bool(row["supports_5pct_claim"].iloc[0])
    assert float(row["ci_hi"].iloc[0]) < PROTOCOL.fa_budget


def test_cross_fitting_is_recorded_as_rejected():
    assert PROTOCOL.protocol == "three_fold_holdout"
    source = (REPO_ROOT / "derail" / "monitor" /
              "fusion_protocol.py").read_text("utf-8")
    assert "Cross-fitting was tried and rejected" in source, (
        "the record must carry why cross-fitting was not used, not just "
        "which protocol won")


def test_every_cited_evidence_artifact_exists():
    missing = [name for name in PROTOCOL.evidence
               if not (REPO_ROOT / name).exists()]
    assert not missing, f"protocol cites artifacts that do not exist: {missing}"


def test_the_admission_rule_is_the_frozen_one():
    assert PROTOCOL.healthy_admission == "task_verifier_must_pass"
    assert PROTOCOL.exclude_unscoreable_from_calibration is True
    assert PROTOCOL.report_both_fa_denominators is True


@pytest.mark.parametrize("column", ["detection_rate_holm_p",
                                    "scoreable_fa_rate_holm_p"])
def test_the_arm_comparison_reports_corrected_p_values(column):
    """Selection may not rest on an uncorrected p-value: several metrics were
    tested per deployment."""
    paired = pd.read_csv(REPO_ROOT / "results" / "tables" /
                         "fusion_arm_paired.csv")
    assert column in paired.columns
    assert paired["replicates"].min() >= 30, (
        "below this project's replicate floor, where a gain at p = 0.004 has "
        "previously come back at p = 0.948 on a later block")


def test_a_measured_but_unadopted_channel_is_recorded_with_its_reason():
    """A channel that was tried and left out is part of the decision, not a
    footnote. Without this the record would read as if it had never been
    considered."""
    assert PROTOCOL.deferred_candidate_channels == ("c",)
    assert not set(PROTOCOL.deferred_candidate_channels) & set(PROTOCOL.channels)
    source = (REPO_ROOT / "derail" / "monitor" /
              "fusion_protocol.py").read_text("utf-8")
    assert "Measured and NOT adopted" in source
    assert "Phase-6 freeze" in source, (
        "the record must say what would settle the question, not only that "
        "it is open")


def test_the_deferred_channel_claim_matches_its_evidence():
    """The reason given for deferring is a measurement, so it has to agree
    with the committed table it rests on: the other classes must come back at
    the two-channel threshold, or the stated cause is wrong."""
    allocation = pd.read_csv(REPO_ROOT / "results" / "tables" /
                             "goal_channel_threshold_allocation.csv")
    detection = [c for c in allocation.columns if c.startswith("det_")]
    others = [c for c in detection if c != "det_goal_drift"]
    assert others, "no non-drift classes in the allocation table to check"
    for dataset, group in allocation.groupby("dataset"):
        means = group.groupby("condition")[detection].mean()
        if "det_goal_drift" not in means.columns or len(means) < 3:
            continue
        base = means.loc["e+m at its own theta"]
        matched = means.loc["e+m+c at the e+m theta"]
        for column in others:
            # A corpus that injects only goal_drift has no other class to
            # check; that is a property of the corpus, not a pass.
            if pd.isna(base[column]) or pd.isna(matched[column]):
                continue
            assert matched[column] >= base[column] - 0.05, (
                f"{dataset}: at a matched threshold the goal channel still "
                f"costs {column} ({matched[column]:.3f} vs {base[column]:.3f}), "
                f"so the deferral reason -- budget, not masking -- is wrong")
        assert matched["det_goal_drift"] > base["det_goal_drift"], (
            f"{dataset}: the goal channel adds no drift detection even at a "
            f"matched threshold, so there is nothing to defer")


def test_the_final_calibration_floor_delivers_the_bound_it_claims() -> None:
    """Recompute the bound rather than trusting the predeclared number."""
    from scipy import stats

    n = PROTOCOL.min_final_calibration_episodes
    conf = PROTOCOL.final_calibration_confidence
    budget = PROTOCOL.fa_budget

    qualifying = [k for k in range(1, n + 1)
                  if stats.beta.ppf(conf, k, n + 1 - k) <= budget]
    assert qualifying, (
        f"no rank qualifies at n={n}: the floor cannot deliver a "
        f"{budget:.0%} bound at {conf:.0%} confidence")

    k = max(qualifying)
    assert stats.beta.ppf(conf, k, n + 1 - k) <= budget
    # And the next rank up must fail, or the floor is not the largest usable k.
    assert stats.beta.ppf(conf, k + 1, n - k) > budget

    nominal = k / (n + 1)
    assert nominal / budget >= 0.70, (
        f"the floor keeps only {nominal / budget:.0%} of the budget usable; "
        "a larger fold was predeclared for a reason")


def test_the_development_floor_cannot_support_the_final_claim() -> None:
    """The development floor must not be mistaken for the final one."""
    from scipy import stats

    n = PROTOCOL.min_threshold_episodes
    conf = PROTOCOL.final_calibration_confidence
    budget = PROTOCOL.fa_budget

    assert PROTOCOL.min_final_calibration_episodes > n
    nominal_rank = int(budget * n)
    realized_upper = stats.beta.ppf(conf, nominal_rank, n + 1 - nominal_rank)
    assert realized_upper > budget, (
        "if the development floor already bounded the budget there would be "
        "no reason to predeclare a larger one")
