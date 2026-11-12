"""A healthy-constant channel must be salvaged, not floored and not dropped."""
from __future__ import annotations

import numpy as np
import pytest

from derail.common import (CHANNEL_SLICES, D_TOTAL_REL,
                           MONOTONE_BOUNDED_CHANNELS, Episode, Standardizer)
from derail.monitor.normalized_fusion import (_DEGENERATE_CAP,
                                              _DEGENERATE_DEADBAND,
                                              _MIN_DISPERSION, ChannelScale,
                                              NormalizedFusionMonitor)


def _scale(full=2.0, baseline=0.0):
    return ChannelScale(median=0.0, dispersion=0.0, n_steps=200,
                        estimator="degenerate", mode="one_sided",
                        baseline=baseline, full_scale=full)


def test_the_healthy_constant_scores_exactly_null() -> None:
    s = _scale()
    assert s.to_common_scale(0.0) == 0.0
    assert s.to_common_scale(-5.0) == 0.0, "below the baseline is not anomalous"


def test_tiny_benign_variation_cannot_score() -> None:
    """Without a deadband a 1e-12 wobble would read as a real departure."""
    s = _scale()
    for wobble in (1e-15, 1e-12, _MIN_DISPERSION):
        assert s.to_common_scale(wobble) == 0.0


def test_a_meaningful_departure_is_anomalous_and_monotone() -> None:
    s = _scale()
    a, b, c = (s.to_common_scale(v) for v in (0.5, 1.0, 4.0))
    assert 0.0 < a < b < c


def test_the_transform_is_bounded_by_the_declared_cap() -> None:
    """A degenerate channel may win a fusion; it may not annihilate it."""
    s = _scale()
    assert s.to_common_scale(1e9) <= _DEGENERATE_CAP
    assert s.to_common_scale(1e300) <= _DEGENERATE_CAP
    assert np.isfinite(s.to_common_scale(1e300))


def test_a_channel_with_no_declared_bound_stays_excluded() -> None:
    s = ChannelScale(median=0.0, dispersion=0.0, n_steps=10,
                     estimator="degenerate", mode="excluded")
    assert not s.usable
    with pytest.raises(ValueError, match="declares no bounded monotone"):
        s.to_common_scale(1.0)


def test_only_channels_that_declare_bounds_are_eligible() -> None:
    """Direction and scale come from the telemetry definition, not the data."""
    assert set(MONOTONE_BOUNDED_CHANNELS) == {"c", "r"}
    for name, (lo, hi) in MONOTONE_BOUNDED_CHANNELS.items():
        assert name in CHANNEL_SLICES and hi > lo
    assert "g" not in MONOTONE_BOUNDED_CHANNELS, (
        "g declares an orientation but no bound, so it has no admissible "
        "full scale and must not be given an invented one")


def _episodes(rng, n, T=8, width=D_TOTAL_REL, quiet_channel="r"):
    sl = CHANNEL_SLICES[quiet_channel]
    out = []
    for i in range(n):
        X = rng.normal(0.0, 1.0, (T, width))
        X[:, sl] = 0.0                       # constant on healthy
        out.append(Episode(X=X, episode_id=f"h{i}", is_healthy=True,
                           failure_class=None, tau=None, t_fail=None,
                           severity=None))
    return out


def test_a_healthy_constant_channel_is_built_not_refused() -> None:
    rng = np.random.default_rng(0)
    fit, scale = _episodes(rng, 30), _episodes(rng, 20)
    mon = NormalizedFusionMonitor(Standardizer().fit(fit), channels=("r",),
                                  statistic="rate", seed=0)
    mon.fit(fit)
    mon.fit_channel_scales(scale)            # previously raised ValueError

    s = mon.scales[0]
    assert s.mode == "one_sided" and s.usable
    assert s.full_scale > _MIN_DISPERSION
    assert mon.active_channels == ("r",)


def test_every_healthy_step_of_the_scale_fold_scores_zero() -> None:
    """The baseline is the top of the observed healthy null, by construction."""
    rng = np.random.default_rng(1)
    fit, scale = _episodes(rng, 30), _episodes(rng, 20)
    mon = NormalizedFusionMonitor(Standardizer().fit(fit), channels=("r",),
                                  statistic="rate", seed=0)
    mon.fit(fit)
    mon.fit_channel_scales(scale)
    for ep in scale:
        mon.start_episode()
        assert max(mon.score_step(ep.X[t]) for t in range(ep.T)) == 0.0


def test_departure_in_the_declared_direction_alarms() -> None:
    rng = np.random.default_rng(2)
    fit, scale = _episodes(rng, 30), _episodes(rng, 20)
    mon = NormalizedFusionMonitor(Standardizer().fit(fit), channels=("r",),
                                  statistic="rate", seed=0)
    mon.fit(fit)
    mon.fit_channel_scales(scale)

    bad = _episodes(rng, 1)[0]
    bad.X[4:, CHANNEL_SLICES["r"]] = 1.0     # saturated, the declared maximum
    mon.start_episode()
    assert max(mon.score_step(bad.X[t]) for t in range(bad.T)) > 0.0


def test_a_non_degenerate_channel_is_untouched() -> None:
    """The ordinary path must keep centring and dividing by the spread."""
    rng = np.random.default_rng(3)
    fit, scale = _episodes(rng, 30), _episodes(rng, 20)
    mon = NormalizedFusionMonitor(Standardizer().fit(fit), channels=("e",),
                                  statistic="rate", seed=0)
    mon.fit(fit)
    mon.fit_channel_scales(scale)
    s = mon.scales[0]
    assert s.mode == "scaled" and s.estimator in ("iqr", "sd")
    assert s.to_common_scale(s.median) == 0.0
    assert s.to_common_scale(s.median + s.dispersion) == pytest.approx(1.0)


def test_all_channels_degenerate_and_undeclared_is_still_refused() -> None:
    rng = np.random.default_rng(4)
    fit, scale = _episodes(rng, 30), _episodes(rng, 20)
    for ep in fit + scale:
        ep.X[:, CHANNEL_SLICES["g"]] = 0.0
    mon = NormalizedFusionMonitor(Standardizer().fit(fit), channels=("g",),
                                  statistic="rate", seed=0)
    mon.fit(fit)
    with pytest.raises(ValueError, match="none declares bounded monotone"):
        mon.fit_channel_scales(scale)


def test_the_probe_leaves_no_state_behind() -> None:
    """Measuring the full scale must not disturb the monitor it measures."""
    rng = np.random.default_rng(5)
    fit, scale = _episodes(rng, 30), _episodes(rng, 20)
    probe_ep = _episodes(rng, 1)[0]

    def scores():
        mon = NormalizedFusionMonitor(Standardizer().fit(fit), channels=("r",),
                                      statistic="rate", seed=0)
        mon.fit(fit)
        mon.fit_channel_scales(scale)
        mon.start_episode()
        return [mon.score_step(probe_ep.X[t]) for t in range(probe_ep.T)]

    assert scores() == scores()


def test_the_deadband_is_relative_to_the_declared_scale() -> None:
    small, large = _scale(full=1e-3), _scale(full=1e3)
    assert small.to_common_scale(1e-3 * _DEGENERATE_DEADBAND * 0.5) == 0.0
    assert large.to_common_scale(1e3 * _DEGENERATE_DEADBAND * 0.5) == 0.0
    assert large.to_common_scale(1e3 * _DEGENERATE_DEADBAND * 100) > 0.0
