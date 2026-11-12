"""Properties the deviation monitor must hold, independent of any corpus."""
from __future__ import annotations

import numpy as np
import pytest

from derail.common import D_TOTAL_GOAL
from derail.monitor.deviation import DeviationMonitor
from derail.monitor.esn import _WASHOUT


class _Ep:
    def __init__(self, X):
        self.X = np.asarray(X, float)
        self.T = self.X.shape[0]


def _healthy(rng, n=40, T=8, width=D_TOTAL_GOAL):
    return [_Ep(rng.normal(0.0, 1.0, (T, width))) for _ in range(n)]


def _fit(rng, **kw):
    healthy = _healthy(rng)
    return DeviationMonitor(**kw).fit(healthy), healthy


def test_a_low_excursion_scores_as_high_as_a_high_one() -> None:
    """The point of the statistic: both tails are anomalous."""
    rng = np.random.default_rng(0)
    mon, _ = _fit(rng, statistic="step", combiner="max")
    d = mon.dims[0]

    base = np.zeros((6, D_TOTAL_GOAL))
    high, low = base.copy(), base.copy()
    high[-1, d] = 8.0
    low[-1, d] = -8.0
    s_high = mon.episode_scores(_Ep(high))[-1]
    s_low = mon.episode_scores(_Ep(low))[-1]

    assert s_high > 1.0 and s_low > 1.0
    assert s_low == pytest.approx(s_high, rel=0.35), (
        "a symmetric null must score the two tails alike")


def test_a_skewed_null_does_not_hide_the_thin_tail() -> None:
    """One shared scale lets the fat tail set the divisor; per-tail does not."""
    rng = np.random.default_rng(1)
    width = D_TOTAL_GOAL
    d = DeviationMonitor().dims[0]
    healthy = []
    for _ in range(60):
        X = np.zeros((8, width))
        X[:, d] = rng.exponential(1.0, 8)          # right-skewed, min 0
        healthy.append(_Ep(X))
    mon = DeviationMonitor(statistic="step", combiner="max").fit(healthy)

    centre, s_lo, s_hi = mon._scales[("level", d)]
    assert s_hi > s_lo, "the exponential null must have a fatter right tail"

    probe = np.zeros((6, width))
    probe[:, d] = rng.exponential(1.0, 6)
    probe[-1, d] = 0.0                              # into the thin left tail
    assert mon.episode_scores(_Ep(probe))[-1] > 1.0


def test_the_score_at_t_cannot_depend_on_a_later_step() -> None:
    rng = np.random.default_rng(2)
    mon, _ = _fit(rng, statistic="cusum")
    X = rng.normal(0, 1, (9, D_TOTAL_GOAL))
    first = mon.episode_scores(_Ep(X))

    changed = X.copy()
    changed[6:] += 50.0
    second = mon.episode_scores(_Ep(changed))
    assert np.allclose(first[:6], second[:6], equal_nan=True)


def test_the_washout_steps_never_alarm() -> None:
    rng = np.random.default_rng(3)
    mon, _ = _fit(rng, statistic="step")
    X = np.full((6, D_TOTAL_GOAL), 40.0)
    scores = mon.episode_scores(_Ep(X))
    assert np.all(np.isneginf(scores[:_WASHOUT]))
    assert np.all(np.isfinite(scores[_WASHOUT:]))


def test_a_dimension_with_no_healthy_spread_is_excluded() -> None:
    rng = np.random.default_rng(4)
    healthy = _healthy(rng)
    d = DeviationMonitor().dims[0]
    for ep in healthy:
        ep.X[:, d] = 3.0
    mon = DeviationMonitor(statistic="step", combiner="max").fit(healthy)
    assert mon._scales[("level", d)] is None, (
        "a constant dim carries no evidence and must not be scored")


def test_the_spread_feature_sees_a_collapse_in_variability() -> None:
    """A run that goes unusually uniform is anomalous even when never extreme."""
    rng = np.random.default_rng(5)
    healthy = _healthy(rng, n=60, T=10)
    mon = DeviationMonitor(statistic="step", combiner="max",
                           feature="spread").fit(healthy)

    flat = _Ep(np.tile(rng.normal(0, 1, (1, D_TOTAL_GOAL)), (10, 1)))
    varied = _Ep(rng.normal(0, 1, (10, D_TOTAL_GOAL)))
    assert float(np.max(mon.episode_scores(flat)[_WASHOUT:])) > \
        float(np.max(mon.episode_scores(varied)[_WASHOUT:]))


def test_fitting_uses_healthy_episodes_only_and_no_labels() -> None:
    """`fit` takes episodes and reads `X` alone — no class, tau or framework."""
    rng = np.random.default_rng(6)
    healthy = _healthy(rng)
    mon = DeviationMonitor().fit(healthy)
    assert mon._scales, "nothing was fitted"
    for ep in healthy:
        assert not hasattr(ep, "failure_class") and not hasattr(ep, "tau")
