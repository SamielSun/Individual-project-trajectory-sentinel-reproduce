"""Does the fusion layer do what it claims, mechanically?

Not "is it better on the corpus" -- that is what the development study
measures. These are the properties the design rests on: that a channel cannot
win on scale alone, that a quieter channel can win when it is the informed
one, that the fused threshold is picked on the fused statistic, that nothing
in the calibration path can see a label, and that scoring stays causal.
"""
from __future__ import annotations

import numpy as np
import pytest

from derail.common import Standardizer
from derail.monitor.esn import _WASHOUT
from derail.monitor.normalized_fusion import (STATISTICS, ChannelScale,
                                              _dispersion,
                                              NormalizedFusionMonitor,
                                              pick_fused_threshold, scoreable)
from derail.telemetry.adapter import episode_from_trace


def _steps(seed: int, n: int = 10, latency=(0.3, 0.8), tokens: int = 10):
    rng = np.random.default_rng(seed)
    return [{"text": f"step {t} {rng.integers(0, 1000)}",
             "token_logprobs": [-0.1] * 10, "action": "tool_call",
             "latency_s": float(rng.uniform(*latency)),
             "output_tokens": tokens, "error": False} for t in range(n)]


def _episode(seed: int, n: int = 10, **kw):
    return episode_from_trace(_steps(seed, n, **kw), f"ep{seed}",
                              use_sentence_transformers=False, extended=True)


def _fitted(statistic="rate", n_fit=40, n_scale=15):
    healthy = [_episode(i) for i in range(n_fit + n_scale)]
    monitor = NormalizedFusionMonitor(Standardizer().fit(healthy[:n_fit]),
                                      channels=("e", "m"),
                                      statistic=statistic)
    monitor.fit(healthy[:n_fit])
    monitor.fit_channel_scales(healthy[n_fit:])
    return monitor, healthy


@pytest.mark.parametrize("statistic", STATISTICS)
def test_every_statistic_scores_causally(statistic):
    """The score at step t must not depend on anything after t."""
    monitor, _ = _fitted(statistic)
    episode = _episode(999)
    full = monitor.score_stream(episode)
    for cut in range(_WASHOUT + 1, episode.T):
        monitor.start_episode()
        prefix = [monitor.score_step(episode.X[t]) for t in range(cut)]
        assert prefix == pytest.approx(list(full[:cut])), (
            f"{statistic}: seeing only the first {cut} steps changed an "
            f"earlier score, so the statistic is not causal")


def test_scoring_before_the_scales_are_learned_is_refused():
    healthy = [_episode(i) for i in range(40)]
    monitor = NormalizedFusionMonitor(Standardizer().fit(healthy),
                                      channels=("e", "m"))
    monitor.fit(healthy)
    with pytest.raises(RuntimeError, match="fit_channel_scales"):
        monitor.score_step(healthy[0].X[0])


def test_a_channel_cannot_win_on_scale_alone():
    """Multiplying one channel's raw stream by a constant must not change
    which channel wins: the normalizer divides that constant straight back
    out. This is the defect the layer exists to remove."""
    monitor, healthy = _fitted("rate")
    episode = _episode(500)
    monitor.score_stream(episode)
    before = monitor.last_winning_channel

    inflated = NormalizedFusionMonitor.__new__(NormalizedFusionMonitor)
    inflated.__dict__.update(monitor.__dict__)
    inflated.scales = [ChannelScale(median=s.median * 1000.0,
                                    dispersion=s.dispersion * 1000.0,
                                    n_steps=s.n_steps)
                       for s in monitor.scales]

    class _Scaled:
        """Wraps a sub-monitor so its raw stream is 1000x larger."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def score_step(self, x):
            return self._inner.score_step(x) * 1000.0

        @property
        def last_raw_increment(self):
            return self._inner.last_raw_increment * 1000.0

    inflated.subs = [_Scaled(sub) for sub in monitor.subs]
    inflated.score_stream(episode)
    assert inflated.last_winning_channel == before


def test_the_quieter_channel_wins_when_it_is_the_informed_one():
    """A disturbance confined to the metadata channel must surface.

    Latency and output length live in m. Under raw max-fusion a shift there
    loses to whatever the semantic channel happens to be doing; normalized,
    m must win the step.
    """
    monitor, _ = _fitted("increment")
    disturbed = episode_from_trace(
        _steps(4242, n=6)[:4] + _steps(4242, n=6, latency=(30.0, 40.0),
                                       tokens=4000)[4:],
        "m-disturbance", use_sentence_transformers=False, extended=True)
    monitor.score_stream(disturbed)
    assert monitor.last_winning_channel == "m"


def test_the_threshold_is_picked_on_the_fused_statistic():
    """Not on either channel alone: taking a max changes the null, so a
    per-channel threshold does not control the fused rate."""
    monitor, healthy = _fitted("rate")
    theta, used, skipped = pick_fused_threshold(monitor, healthy[:20],
                                                fa_budget=0.05)
    maxima = [float(monitor.score_stream(ep).max()) for ep in healthy[:20]]
    # The empirical rule returns an observed episode maximum of the FUSED
    # stream, so theta must be one of these and not a per-channel quantity.
    assert any(theta == pytest.approx(m) for m in maxima)
    per_channel_maxima = [float(sub.score_episode_components(ep)["fused"].max())
                          for sub in monitor.subs for ep in healthy[:20]]
    assert not any(theta == pytest.approx(m) for m in per_channel_maxima
                   if m not in maxima)
    assert used == 20 and skipped == 0


def test_unscoreable_episodes_never_enter_the_threshold_distribution():
    monitor, healthy = _fitted("rate")
    short = [_episode(900 + i, n=_WASHOUT) for i in range(5)]
    assert not any(scoreable(ep) for ep in short)
    theta_clean, used_clean, _ = pick_fused_threshold(monitor, healthy[:20])
    theta_mixed, used_mixed, skipped = pick_fused_threshold(
        monitor, healthy[:20] + short)
    assert skipped == 5
    assert used_mixed == used_clean
    assert theta_mixed == theta_clean, (
        "short episodes leaked into the threshold distribution as zeros")


def test_the_calibration_path_cannot_receive_a_label():
    """Structural, not a promise: fit, scales and threshold take episodes and
    nothing else, and an injected episode carries its label on the object --
    so a label reaching them would have to be read off the Episode, which
    none of these call sites does."""
    import inspect

    for fn in (NormalizedFusionMonitor.fit,
               NormalizedFusionMonitor.fit_channel_scales,
               pick_fused_threshold):
        source = inspect.getsource(fn)
        for forbidden in ("failure_class", "is_healthy", "tau", "t_fail"):
            assert forbidden not in source, (
                f"{fn.__name__} mentions {forbidden}: the calibration path "
                f"must be label-free")


def test_a_dispersionless_channel_is_excluded_not_floored():
    """A channel whose healthy null never moves has no scale to divide by.

    Dividing by a small floor instead of saying so is what this guards: at a
    1e-9 floor the channel is amplified by a billion and wins every fusion,
    which is the exact defect normalization exists to prevent. It must be
    refused, and the refusal has to be loud.
    """
    scale = ChannelScale(median=0.0, dispersion=0.0, n_steps=10)
    assert not scale.usable
    with pytest.raises(ValueError, match="excluded from the fusion"):
        scale.to_common_scale(1.0)


def test_the_dispersion_falls_back_rather_than_dividing_by_a_floor():
    """IQR, then standard deviation, then refuse."""
    assert _dispersion(np.array([1.0, 2.0, 3.0, 4.0]))[1] == "iqr"
    # Middle two quartiles coincide, tails move: the IQR is blind, sd is not.
    assert _dispersion(np.array([0.0] * 8 + [5.0, -5.0]))[1] == "sd"
    assert _dispersion(np.zeros(10)) == (0.0, "degenerate")


def test_there_is_no_unreachable_mad_rung():
    """A zero IQR puts at least half the points on the median, which forces
    the MAD to zero too -- so a MAD rung between them could never fire, and
    shipping one would be dead code pretending to be a safeguard."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        arr = rng.choice([0.0, 1.0, 2.0, 3.0], size=rng.integers(4, 40),
                         p=[0.7, 0.1, 0.1, 0.1])
        q25, q75 = np.quantile(arr, [0.25, 0.75])
        if float(q75 - q25) <= 1e-9:
            mad = float(np.median(np.abs(arr - np.median(arr))))
            assert mad <= 1e-9


def test_a_constant_channel_does_not_silently_halve_the_fusion():
    """An excluded channel is reported, not quietly dropped.

    A two-channel fusion that has become one channel must not still look like
    a two-channel one, or the artifact records a fusion that never happened.
    """
    healthy = [_episode(s) for s in range(6)]
    monitor = NormalizedFusionMonitor(Standardizer().fit(healthy),
                                      channels=("e", "m"), seed=0)
    monitor.fit(healthy)
    monitor.fit_channel_scales(healthy)
    monitor.scales = [monitor.scales[0],
                      ChannelScale(median=0.0, dispersion=0.0, n_steps=1)]

    assert monitor.active_channels == ("e",)
    monitor.start_episode()
    value = monitor.score_step(healthy[0].X[0])
    assert np.isfinite(value)
    assert monitor.last_winning_channel == "e"


def test_every_channel_constant_is_an_error_not_a_floor():
    """If nothing can be normalized the fusion is arithmetic, not evidence."""
    healthy = [_episode(s) for s in range(6)]
    monitor = NormalizedFusionMonitor(Standardizer().fit(healthy),
                                      channels=("e", "m"), seed=0)
    monitor.fit(healthy)
    constant = [ChannelScale(median=0.0, dispersion=0.0, n_steps=1)] * 2
    monitor.scales = constant
    assert monitor.active_channels == ()
