"""An inert branch must not cost a useful one anything.

Controlled synthetic scores, so the property is checked against arithmetic
rather than against a corpus. `test_averaging_dilutes...` is the control: it
shows the same construction failing under the mean fusion these branches used
to go through, which is what makes the passing cases meaningful.
"""
from __future__ import annotations

import numpy as np
import pytest

from derail.monitor.evidence_fusion import (_TIE_TOL, COMBINERS,
                                            EvidenceUnion, TailEvidence)


def _null(seed=0, n=4000):
    return np.random.default_rng(seed).normal(size=n)


# ------------------------------------------------------------ the evidence

def test_at_or_below_the_healthy_floor_the_evidence_is_exactly_zero() -> None:
    cal = TailEvidence.fit(_null())
    assert cal.evidence(cal.null[0]) == 0.0
    assert cal.evidence(cal.null[0] - 1e6) == 0.0


def test_evidence_is_monotone_in_the_branch_statistic() -> None:
    cal = TailEvidence.fit(_null())
    grid = np.linspace(-8.0, 20.0, 2000)
    assert np.all(np.diff(cal.evidence(grid)) >= -1e-12)


def test_evidence_is_bounded_by_the_fold_resolution_plus_the_extension() -> None:
    cal = TailEvidence.fit(_null())
    top = 2 * cal.ceiling
    for v in (10.0, 1e6, 1e300):
        assert 0.0 <= cal.evidence(v) <= top + 1e-9
    assert np.isfinite(cal.evidence(1e300))


def test_a_float_wobble_on_a_constant_null_is_not_a_departure() -> None:
    cal = TailEvidence.fit(np.zeros(500))
    assert cal.resolution_limited
    for wobble in (1e-18, 1e-15, _TIE_TOL * 0.5):
        assert cal.evidence(wobble) == 0.0
    assert cal.evidence(1.0) == pytest.approx(cal.ceiling)


def test_an_unscoreable_step_stays_absent_rather_than_becoming_quiet() -> None:
    cal = TailEvidence.fit(_null())
    assert cal.evidence(-np.inf) == -np.inf
    assert cal.evidence(np.nan) == -np.inf


def test_an_empty_null_is_refused_not_defaulted() -> None:
    with pytest.raises(ValueError, match="no finite healthy value"):
        TailEvidence.fit([np.inf, np.nan])


# ----------------------------------------------------------- the invariant

def _episode_max(union, streams_per_ep):
    return np.array([float(np.nanmax(union.combine(s)))
                     for s in streams_per_ep])


def test_a_deterministically_inert_branch_changes_nothing_at_all() -> None:
    """The property the whole module exists for, at the score level."""
    rng = np.random.default_rng(1)
    null_a, quiet = _null(2), np.zeros(4000)
    alone = EvidenceUnion(("a",), "max").fit({"a": [null_a]})
    both = EvidenceUnion(("a", "b"), "max").fit({"a": [null_a], "b": [quiet]})
    for _ in range(20):
        probe = rng.normal(size=40)
        s = {"a": probe, "b": np.zeros(40)}
        assert np.array_equal(both.combine(s), alone.combine({"a": probe}))


def test_a_deterministically_inert_branch_changes_no_alarm(monkeypatch) -> None:
    """And therefore no threshold, no false alarm and no detection."""
    rng = np.random.default_rng(3)
    null_a, quiet = _null(4), np.zeros(4000)
    alone = EvidenceUnion(("a",), "max").fit({"a": [null_a]})
    both = EvidenceUnion(("a", "b"), "max").fit({"a": [null_a], "b": [quiet]})

    healthy = [rng.normal(size=30) for _ in range(60)]
    failing = [rng.normal(size=30) + 3.0 for _ in range(30)]

    def rates(union, keys):
        def streams(p):
            return {k: (p if k == "a" else np.zeros(p.size)) for k in keys}
        cal = sorted((float(union.combine(streams(p)).max()) for p in healthy),
                     reverse=True)
        theta = cal[int(np.floor(0.05 * len(cal))) - 1]
        det = np.mean([union.combine(streams(p)).max() >= theta
                       for p in failing])
        return theta, det

    assert rates(both, ("a", "b")) == rates(alone, ("a",))


def test_averaging_dilutes_where_the_union_does_not() -> None:
    """The control: the same inert branch, under the fusion that was there."""
    rng = np.random.default_rng(5)
    healthy = [rng.normal(size=30) for _ in range(200)]
    failing = [rng.normal(size=30) + 2.5 for _ in range(200)]

    def detection(k_inert, reduce_):
        def fuse(p):
            cols = [p] + [rng.normal(size=p.size) for _ in range(k_inert)]
            return reduce_(np.stack(cols, axis=1))
        cal = sorted((float(fuse(p).max()) for p in healthy), reverse=True)
        theta = cal[int(np.floor(0.05 * len(cal))) - 1]
        return float(np.mean([fuse(p).max() >= theta for p in failing]))

    mean_alone = detection(0, lambda A: A.mean(axis=1))
    mean_diluted = detection(3, lambda A: A.mean(axis=1))
    assert mean_diluted < mean_alone - 0.05, (
        "the control must actually show dilution, or the invariant tests "
        "above are not testing anything")


def test_the_union_never_scores_below_any_branch_it_contains() -> None:
    """Holds for a noisy-inert branch too; only the threshold can move."""
    rng = np.random.default_rng(7)
    cals = {"a": _null(8), "b": _null(9) * 3.0}
    both = EvidenceUnion(("a", "b"), "max").fit({k: [v] for k, v in cals.items()})
    alone = EvidenceUnion(("a",), "max").fit({"a": [cals["a"]]})
    for _ in range(20):
        s = {"a": rng.normal(size=50), "b": rng.normal(size=50) * 3.0}
        assert np.all(both.combine(s) >= alone.combine({"a": s["a"]}) - 1e-12)


# ------------------------------------------------------------- the plumbing

@pytest.mark.parametrize("combiner", COMBINERS)
def test_every_combiner_is_monotone_in_each_branch(combiner) -> None:
    rng = np.random.default_rng(11)
    union = EvidenceUnion(("a", "b"), combiner).fit(
        {"a": [_null(12)], "b": [_null(13)]})
    base = {"a": rng.normal(size=200), "b": rng.normal(size=200)}
    lifted = {"a": base["a"] + 0.5, "b": base["b"]}
    assert np.all(union.combine(lifted) >= union.combine(base) - 1e-12)


@pytest.mark.parametrize("combiner", COMBINERS)
def test_a_step_no_branch_could_score_never_alarms(combiner) -> None:
    union = EvidenceUnion(("a", "b"), combiner).fit(
        {"a": [_null(14)], "b": [_null(15)]})
    s = {"a": np.full(4, -np.inf), "b": np.full(4, -np.inf)}
    assert np.all(union.combine(s) == -np.inf)


def test_an_unknown_combiner_is_refused() -> None:
    with pytest.raises(ValueError, match="combiner must be one of"):
        EvidenceUnion(("a",), "median")
    with pytest.raises(ValueError, match="at least one branch"):
        EvidenceUnion((), "max")


def test_scoring_before_calibrating_is_refused() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\) must run before"):
        EvidenceUnion(("a",)).combine({"a": np.zeros(3)})


def test_evidence_of_a_step_does_not_depend_on_the_steps_around_it() -> None:
    """Prefix determinism: the map is per-step, so a prefix cannot be revised."""
    rng = np.random.default_rng(17)
    union = EvidenceUnion(("a",), "max").fit({"a": [_null(18)]})
    full = rng.normal(size=60)
    whole = union.combine({"a": full})
    for cut in (5, 17, 40):
        assert np.array_equal(union.combine({"a": full[:cut]}), whole[:cut])


def test_the_null_is_a_multiset_so_episode_identity_cannot_reach_it() -> None:
    """Label invariance: only the healthy VALUES enter, in any order."""
    rng = np.random.default_rng(19)
    parts = [rng.normal(size=50) for _ in range(8)]
    a = EvidenceUnion(("x",), "max").fit({"x": parts})
    b = EvidenceUnion(("x",), "max").fit({"x": parts[::-1]})
    probe = rng.normal(size=30)
    assert np.array_equal(a.combine({"x": probe}), b.combine({"x": probe}))


def test_a_branch_with_no_estimable_tail_is_reported_as_such() -> None:
    union = EvidenceUnion(("a", "b"), "max").fit(
        {"a": [_null(21)], "b": [np.zeros(300)]})
    assert union.resolution_limited == ("b",)


def test_a_short_fold_still_gets_a_tail_rather_than_a_flat_ceiling() -> None:
    """Regression: at the ~100 steps a real scale fold holds, counting
    exceedances from the 95th percentile left 6 of them, every branch fell
    back to a flat ceiling, and healthy episodes tied there often enough to
    take the false-alarm rate to twice the budget."""
    cal = TailEvidence.fit(np.random.default_rng(23).normal(size=106))
    assert not cal.resolution_limited
    above = cal.evidence(np.array([cal.null[-1] + d
                                   for d in (0.1, 0.4, 0.9, 1.6)]))
    assert len(set(above.tolist())) == 4, "departures must not all tie"
    assert np.all(np.diff(above) > 0)


def test_the_extreme_tail_keeps_ranking_instead_of_piling_on_a_cap() -> None:
    """Regression: a linear continuation under a hard cap moved the tie from
    the ceiling to the cap, healthy episodes piled up there, and the union's
    false-alarm rate stayed at 6-10% against a 5% budget."""
    cal = TailEvidence.fit(np.random.default_rng(29).normal(size=190))
    far = cal.evidence(cal.null[-1] + np.array([1.0, 4.0, 16.0, 64.0, 256.0]))
    assert len(set(far.tolist())) == 5
    assert np.all(np.diff(far) > 0)
    assert far[-1] <= 2 * cal.ceiling + 1e-9


@pytest.mark.parametrize("combiner", COMBINERS)
def test_every_combiner_ignores_a_deterministically_inert_branch(combiner):
    """Non-dilution is not a property of max alone: a branch pinned at its
    healthy floor contributes 0, which sum and top2 also absorb, and the
    omnibus inherits it because its second level sees identical streams."""
    rng = np.random.default_rng(31)
    a, b, quiet = _null(32), _null(33), np.zeros(4000)
    two = EvidenceUnion(("a", "b"), combiner).fit({"a": [a], "b": [b]})
    three = EvidenceUnion(("a", "b", "c"), combiner).fit(
        {"a": [a], "b": [b], "c": [quiet]})
    for _ in range(10):
        s = {"a": rng.normal(size=40), "b": rng.normal(size=40)}
        got = three.combine({**s, "c": np.zeros(40)})
        assert np.allclose(got, two.combine(s), rtol=0, atol=1e-12)


def test_the_omnibus_fires_on_concentrated_and_on_diffuse_evidence() -> None:
    """The reason it exists: max reads one hard branch and misses a broad
    lift, sum does the reverse, and the failure decides which -- not the
    corpus, so neither rule can be chosen in advance."""
    rng = np.random.default_rng(37)
    keys = tuple("abcd")
    nulls = {k: rng.normal(size=6000) for k in keys}
    fitted = {c: EvidenceUnion(keys, c).fit({k: [v] for k, v in nulls.items()})
              for c in ("max", "sum", "omnibus")}

    healthy = {k: rng.normal(size=6000) for k in keys}
    concentrated = {k: rng.normal(size=6000) + (3.0 if k == "a" else 0.0)
                    for k in keys}
    diffuse = {k: rng.normal(size=6000) + 0.9 for k in keys}

    def power(rule, alt):
        theta = np.quantile(fitted[rule].combine(healthy), 0.95)
        return float(np.mean(fitted[rule].combine(alt) >= theta))

    # Each fixed rule is the wrong one in one of the two regimes.
    assert power("max", concentrated) > power("sum", concentrated)
    assert power("sum", diffuse) > power("max", diffuse)
    # The omnibus is never the one stuck with the wrong rule.
    assert power("omnibus", concentrated) > power("sum", concentrated)
    assert power("omnibus", diffuse) > power("max", diffuse)
