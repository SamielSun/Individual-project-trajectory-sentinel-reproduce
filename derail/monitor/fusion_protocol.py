"""The calibration protocol chosen for the post-fix detector.

A decision record, not the final freeze. It fixes how a deployment monitor is
built and thresholded, and why, so that the choice is committed before any
fresh evaluation data exists and cannot quietly follow the results. The
Phase-6 freeze will add configuration hashes and the episode schedule on top
of this.

Every value here was chosen from development data and from the committed
studies named beside it. Nothing was chosen from a final held-out outcome,
because none exists yet.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = REPO_ROOT / "results" / "fusion_protocol_decision.json"


@dataclass(frozen=True)
class FusionProtocol:
    """How a per-deployment monitor is fitted, scaled, fused and thresholded."""

    version: str = "normalized_fusion[e+m]/rate/v2"

    #: Per-channel causal statistic. `rate` divides the running CUSUM by the
    #: steps scored so far. Chosen because the length coupling is in the
    #: statistic, not in the fusion: across trajectory-length bands the raw
    #: and normalized CUSUM arms both run healthy false alarms of
    #: 0.00/0.05/0.25 (langgraph) and 0.06/0.13/0.20 (autogen), while `rate`
    #: is flat. Over 40 paired dev-splits it is the only arm positive on both
    #: deployments, significant on one after Holm correction
    #: (autogen +0.121 detection, Holm p = 0.031) and non-negative on the
    #: other (langgraph +0.027, Holm p = 1.000).
    statistic: str = "rate"

    #: Channels fused. `u` is absent because these runtimes surface no
    #: logprobs, which is a property of the deployment, not a choice.
    channels: tuple[str, ...] = ("e", "m")

    #: Measured and NOT adopted. The goal-consistency channel `c`
    #: (DESIGN.md amendment 8) is the only telemetry that can express "this
    #: is not the task that was asked", and adding it raises detection of
    #: REALIZED goal drift on all three development corpora at no
    #: false-alarm cost (LangGraph 0.117 -> 0.550, AutoGen 0.458 -> 0.708,
    #: long-drift 0.080 -> 0.205; Holm-significant on each). It is not in
    #: `channels` because the same runs lose an equal amount on every other
    #: class, and the cause is arithmetic rather than interference: scoring
    #: the three-channel monitor at the two-channel threshold restores the
    #: other classes exactly AND keeps the goal-drift gain. A fixed budget
    #: split across one more channel is the whole effect. On a threshold
    #: fold of 17-24 episodes the order-statistic floor 1/(n+1) already
    #: exceeds 5%, so one healthy episode sets theta and the split cannot be
    #: measured. Decide at the Phase-6 freeze, on >= 80 threshold episodes.
    deferred_candidate_channels: tuple[str, ...] = ("c",)

    #: Each channel is mapped through the median and IQR of ITS OWN healthy
    #: null before the max. Retained on structural grounds, not for accuracy:
    #: measured over 40 paired splits it moves detection by +0.009 and +0.008
    #: (Holm p = 1.000 on both), so it is NOT an improvement in itself. What
    #: it buys is that the fusion stops being decided by which channel
    #: happens to carry larger numbers -- without it the fused threshold is
    #: identical to the semantic channel's own and the metadata channel is
    #: decorative. Cost is about 2% of per-step scoring time.
    normalize_channels: bool = True

    #: Threshold is picked on the FUSED statistic, never inferred from
    #: per-channel budgets: taking a max changes the null and the channels
    #: are correlated.
    threshold_on: str = "fused_statistic"
    fa_budget: float = 0.05

    #: Three disjoint healthy folds. Cross-fitting was tried and rejected:
    #: out-of-fold scores come from K monitors whose standardizers,
    #: reservoirs and channel scales all differ from the final refitted one,
    #: so an OOF threshold is in the wrong units. Measured, it moved theta
    #: from 8.26 to 13.17 (langgraph, false alarms 0.000 -> 0.429) and from
    #: 89.6 to 786.1 (autogen, detection 0.609 -> 0.000). Cross-fitting stays
    #: valid only where the score scale is fixed across folds.
    protocol: str = "three_fold_holdout"
    min_fit_episodes: int = 140
    min_scale_episodes: int = 40
    min_threshold_episodes: int = 80

    #: Final claim only; the floors above govern development. `fa_budget` is a
    #: nominal rate whose median realization is 5%, so "at most 5%" needs the
    #: largest k with BetaInv(confidence; k, n+1-k) <= fa_budget. At n = 640
    #: that is k = 23: nominal 3.6% under a guaranteed 4.87%.
    #: See results/conditional_calibration_sizing.md.
    final_calibration_rule: str = "distribution_free_one_sided_tolerance_bound"
    final_calibration_confidence: float = 0.95
    min_final_calibration_episodes: int = 640

    #: Untouched, evaluation only. At 300 an
    #: observed 2% false-alarm rate still bounds below 5% (exact 95% CI
    #: 0.007-0.043); at 24 even a zero observation bounds only at 14.2%.
    untouched_test_episodes: int = 300

    #: Healthy admission, frozen before calibration. A run is a healthy
    #: negative only if the task's own predeclared verifier passes it.
    #: Unscoreable episodes are kept and counted, never used to fit, scale or
    #: threshold, and never enter the threshold distribution as zeros.
    healthy_admission: str = "task_verifier_must_pass"
    exclude_unscoreable_from_calibration: bool = True
    report_both_fa_denominators: bool = True

    evidence: tuple[str, ...] = field(default_factory=lambda: (
        "results/fusion_candidates.md",
        "results/tables/fusion_arm_paired.csv",
        "results/tables/fusion_length_strata.csv",
        "results/tables/fusion_channel_wins.csv",
        "results/calibration_sample_size.md",
        "results/healthy_admission_audit.md",
        "results/goal_channel_study.md",
        "results/tables/goal_channel_paired.csv",
        "results/tables/goal_channel_threshold_allocation.csv",
        "results/drift_realization_audit.md",
    ))

    def as_record(self) -> dict:
        record = asdict(self)
        record["channels"] = list(self.channels)
        record["deferred_candidate_channels"] = list(
            self.deferred_candidate_channels)
        record["evidence"] = list(self.evidence)
        return record


PROTOCOL = FusionProtocol()


def write_record() -> Path:
    RECORD_PATH.write_text(
        json.dumps(PROTOCOL.as_record(), indent=2) + "\n", "utf-8")
    return RECORD_PATH


if __name__ == "__main__":
    print(f"[fusion-protocol] wrote {write_record()}")
