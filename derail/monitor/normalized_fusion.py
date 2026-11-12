"""Fuse per-channel detectors on a common healthy-null scale.

`ChannelMaxESNMonitor` takes the max of each channel's CUSUM state. Those
states are not on the same scale -- measured on the framework corpora the
semantic channel wins 93-96% of steps and the fused threshold is identical to
the semantic channel's own, so the metadata channel is silenced by
construction rather than by being uninformative. Two changes here, and they
are separate:

  common scale   each channel's causal stream is mapped through the location
                 and dispersion of THAT channel's own healthy null, so the
                 max compares like with like;
  fused          the max of two calibrated channels is not itself calibrated.
  calibration    Taking a max changes the null, and the channels are
                 correlated, so a 5% budget per channel does not give 5%
                 fused. The threshold is therefore picked on the FUSED
                 statistic, from healthy episodes that set neither the fit
                 nor the scales.

Everything is causal and label-free. The scales and the threshold are frozen
before serving; at serving time step t uses only x_0..x_t. No failure label,
onset or class reaches any of it -- `fit`, `fit_channel_scales` and
`pick_fused_threshold` all take healthy episodes only.

Three per-channel statistics are selectable because they carry different
length behaviour, which is the other half of the problem: `cusum` is the
running CUSUM the shipped monitor uses and grows with the number of scored
steps; `rate` divides it by the steps scored so far; `increment` drops
accumulation and uses the step's own evidence. All three are computable at
step t from x_0..x_t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from derail.common import (CHANNEL_SLICES, Episode,
                           MONOTONE_BOUNDED_CHANNELS, Standardizer)
from derail.evaluation.metrics import pick_threshold
from derail.monitor.esn import _WASHOUT, ESNEnsembleMonitor, OnlineMonitor

#: Per-channel causal statistics this monitor can fuse.
STATISTICS = ("cusum", "rate", "increment")

#: Dispersion floor. Below this a channel's healthy null is a single repeated
#: value and there is no scale to divide by. Dividing by the floor itself is
#: NOT the answer -- it amplifies that channel by 1e9 and hands it every
#: fusion, which is the exact defect this module exists to prevent. It marks
#: the channel unusable instead; see `_dispersion`.
_MIN_DISPERSION = 1e-9

#: A channel whose healthy null is a single value carries no dispersion, but
#: it is not therefore uninformative: if its own telemetry definition declares
#: that every dim is bounded and that higher means more anomalous, then the
#: healthy constant IS the floor and any departure above it is anomalous by
#: construction. Such a channel is scored one-sided against a full scale taken
#: from that declaration rather than from the data.
#:
#: The cap keeps a bounded channel bounded: fully saturated telemetry reads as
#: _DEGENERATE_CAP on the common scale, which is beyond any plausible healthy
#: z-score yet finite, so a degenerate channel can win a fusion but cannot make
#: every other channel irrelevant. Raising it toward infinity would recreate
#: exactly the divide-by-a-floor defect this module exists to prevent.
_DEGENERATE_CAP = 8.0

#: Departures below this fraction of the full scale read as zero. The healthy
#: constant is only constant to within `_MIN_DISPERSION`, and without a
#: deadband a benign 1e-12 wobble would score as a real departure.
_DEGENERATE_DEADBAND = 1e-6

#: The full scale is the statistic's response to ONE maximally anomalous step,
#: measured against the same probe held at the declared floor. A sustained
#: saturation run was tried first and is the wrong unit: it drives the CUSUM
#: far beyond anything a real episode produces, so genuine departures map to a
#: sliver of the cap and the channel can never win a fusion.

def _dispersion(arr: np.ndarray) -> tuple[float, str]:
    """Robust spread of a healthy null, with one fallback.

    The IQR is the first choice and is right whenever the null actually
    varies. It degenerates to exactly zero when the middle two quartiles
    collapse onto one value, which is common here rather than exotic: these
    episodes are short, so a quiet channel's CUSUM sits at 0 for most of the
    steps the scale fold contains.

    The fallback is the standard deviation, which still sees a null whose
    middle is flat but whose tails move. There is deliberately no median
    absolute deviation rung between them: a zero IQR means at least half the
    points already sit on the median, which forces the MAD to zero as well,
    so it could never fire.

    A channel flat under both is genuinely constant on healthy data. It
    carries no scale, so it cannot be put on a common one, and the honest
    thing is to say so -- not to divide by a floor and let it win.
    """
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr > _MIN_DISPERSION:
        return iqr, "iqr"
    sd = float(arr.std())
    if sd > _MIN_DISPERSION:
        return sd, "sd"
    return 0.0, "degenerate"


@dataclass(frozen=True)
class ChannelScale:
    """Location and dispersion of one channel's healthy null.

    Robust statistics rather than mean and standard deviation: these streams
    are heavy-tailed by construction -- a CUSUM's whole purpose is to grow
    without bound when something is wrong -- and a mean would be set by the
    tail it is supposed to measure against.

    `estimator` records which rung of the fallback ladder produced the
    dispersion, so a scale that had to reach past the IQR is visible in the
    frozen artifact instead of being indistinguishable from a clean one.
    """

    median: float
    dispersion: float
    n_steps: int
    estimator: str = "iqr"
    #: "scaled"    the ordinary case: centre and divide by the healthy spread.
    #: "one_sided" healthy-constant, but the channel declares bounded monotone
    #:             dims, so it is scored against a declared full scale.
    #: "excluded"  healthy-constant with no admissible full scale.
    mode: str = "scaled"
    baseline: float = 0.0     # one_sided: the healthy constant, ceiling of null
    full_scale: float = 0.0   # one_sided: statistic at declared saturation

    @property
    def usable(self) -> bool:
        """Can this channel contribute to the fusion at all?"""
        if self.mode == "one_sided":
            return self.full_scale > _MIN_DISPERSION
        if self.mode == "excluded":
            return False
        return self.dispersion > _MIN_DISPERSION

    def to_common_scale(self, value: float) -> float:
        if not self.usable:
            raise ValueError(
                "channel has no healthy dispersion to normalize against and "
                "declares no bounded monotone semantics; it must be excluded "
                "from the fusion, not divided by a floor")
        if self.mode == "one_sided":
            excess = value - self.baseline
            if excess <= _DEGENERATE_DEADBAND * max(self.full_scale, 1e-12):
                return 0.0
            # Saturating, so the declared full scale maps to the cap and no
            # larger reading can exceed it.
            return float(_DEGENERATE_CAP
                         * (1.0 - np.exp(-excess / self.full_scale)))
        return (value - self.median) / self.dispersion


def scoreable(episode: Episode) -> bool:
    """Does the detector get at least one scoring opportunity here?

    An episode shorter than the washout produces no score at any step. It
    must not enter a healthy null as a run of zeros: those are the absence of
    a measurement, not evidence of quiet healthy behaviour, and they drag a
    threshold quantile toward zero.
    """
    return episode.T > _WASHOUT


class NormalizedFusionMonitor(OnlineMonitor):
    """Per-channel ESN-CUSUM detectors fused on a shared healthy-null scale."""

    def __init__(self, standardizer: Standardizer, *,
                 channels: tuple[str, ...] = ("e", "m"),
                 statistic: str = "rate", K: int = 8, seed: int = 0,
                 name: str | None = None, **esn_kwargs) -> None:
        if statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {STATISTICS}, "
                             f"got {statistic!r}")
        self.name = name or f"normalized_fusion[{'+'.join(channels)}]/{statistic}"
        self.channels = tuple(channels)
        self.statistic = statistic
        self.subs = [
            ESNEnsembleMonitor(standardizer, channels=(c,), K=K, cusum=True,
                               seed=seed * 100 + i, **esn_kwargs)
            for i, c in enumerate(self.channels)
        ]
        self.scales: list[ChannelScale] | None = None
        self._i = 0
        self._last_channel_values: list[float] = []

    # ------------------------------------------------------------- fitting

    def fit(self, healthy_episodes: list[Episode]) -> None:
        """Fit each channel's reservoir and readout. Healthy episodes only."""
        for sub in self.subs:
            sub.fit(healthy_episodes)

    def fit_channel_scales(self, healthy_episodes: list[Episode]) -> None:
        """Learn each channel's healthy null, on episodes the fit never saw.

        Fitting the scales on the same episodes that fit the readout would
        measure the residual the readout has already minimised, and every
        channel would look equally quiet.
        """
        usable = [ep for ep in healthy_episodes if scoreable(ep)]
        if not usable:
            raise ValueError(
                "no scoreable healthy episode to learn channel scales from: "
                f"every episode is shorter than the washout ({_WASHOUT})")
        per_channel: list[list[float]] = [[] for _ in self.subs]
        for ep in usable:
            for index, values in enumerate(self._raw_channel_streams(ep)):
                per_channel[index].extend(values[_WASHOUT:])
        width = usable[0].X.shape[1]
        scales = []
        for index, values in enumerate(per_channel):
            arr = np.asarray(values, dtype=float)
            dispersion, estimator = _dispersion(arr)
            if dispersion > _MIN_DISPERSION:
                scales.append(ChannelScale(median=float(np.median(arr)),
                                           dispersion=dispersion,
                                           n_steps=int(arr.size),
                                           estimator=estimator))
                continue
            # Healthy-constant. Salvageable only if the channel's own
            # definition declares bounded monotone dims; the baseline is the
            # top of the observed healthy null, so every healthy step this
            # fold contains scores exactly zero.
            channel = self.channels[index]
            full = self._declared_full_scale(channel, width)
            if full is None or full <= _MIN_DISPERSION:
                scales.append(ChannelScale(median=float(np.median(arr)),
                                           dispersion=dispersion,
                                           n_steps=int(arr.size),
                                           estimator=estimator,
                                           mode="excluded"))
                continue
            scales.append(ChannelScale(median=float(np.median(arr)),
                                       dispersion=dispersion,
                                       n_steps=int(arr.size),
                                       estimator=estimator,
                                       mode="one_sided",
                                       baseline=float(arr.max()),
                                       full_scale=float(full)))
        if not any(scale.usable for scale in scales):
            raise ValueError(
                f"{self.name}: every channel's healthy null is constant on "
                f"the scale fold and none declares bounded monotone dims, so "
                f"none can be normalized. This is a telemetry or fold-size "
                f"problem, not something to floor away -- the fusion would be "
                f"decided by arithmetic rather than evidence.")
        self.scales = scales

    def _declared_full_scale(self, channel: str, width: int) -> float | None:
        """The statistic this channel would read at its declared saturation.

        Configuration only: the channel's dims are driven to the top of the
        range its own telemetry definition declares, and the resulting
        statistic is the unit a departure is measured in. No corpus, no label
        and no outcome enters it. Channels with no declared bound return None
        and stay excluded rather than being given an invented scale.
        """
        bounds = MONOTONE_BOUNDED_CHANNELS.get(channel)
        if bounds is None or channel not in CHANNEL_SLICES:
            return None
        sub = self.subs[self.channels.index(channel)]
        sl = CHANNEL_SLICES[channel]
        if sl.stop > width:
            return None
        low = np.zeros(width, dtype=float)
        low[sl] = bounds[0]
        high = np.zeros(width, dtype=float)
        high[sl] = bounds[1]

        def run(last: np.ndarray) -> float:
            sub.start_episode()
            value = 0.0
            for step in range(_WASHOUT + 1):
                probe = last if step == _WASHOUT else low
                cusum = sub.score_step(probe)
                value = (cusum if self.statistic == "cusum"
                         else cusum / max(1, step - _WASHOUT + 1)
                         if self.statistic == "rate"
                         else sub.last_raw_increment)
            return float(value)

        try:
            return run(high) - run(low)
        finally:
            # The probe must leave no trace: scoring always begins with
            # start_episode, and this restores that contract explicitly.
            sub.start_episode()

    @property
    def active_channels(self) -> tuple[str, ...]:
        """Channels that carry a usable healthy scale, in fusion order.

        A channel whose healthy null is constant is excluded rather than
        normalized. Reporting it here keeps that exclusion visible: a fusion
        that quietly became single-channel should not look like a two-channel
        one.
        """
        if self.scales is None:
            return ()
        return tuple(c for c, s in zip(self.channels, self.scales) if s.usable)

    # ------------------------------------------------------------ scoring

    def start_episode(self) -> None:
        for sub in self.subs:
            sub.start_episode()
        self._i = 0
        self._last_channel_values = []

    def score_step(self, x_t: np.ndarray) -> float:
        if self.scales is None:
            raise RuntimeError(
                f"{self.name}: fit_channel_scales() must be called before "
                f"scoring -- fusing unnormalized channels is the defect this "
                f"monitor exists to fix")
        scored_so_far = max(0, self._i - _WASHOUT + 1)
        values = []
        for sub, scale in zip(self.subs, self.scales):
            cusum = sub.score_step(x_t)
            if self.statistic == "cusum":
                raw = cusum
            elif self.statistic == "rate":
                raw = cusum / max(scored_so_far, 1)
            else:
                raw = sub.last_raw_increment
            # An unusable channel is still stepped -- its reservoir state has
            # to stay in sync -- but it cannot enter the max, because it has
            # no scale that would make its magnitude comparable.
            values.append(scale.to_common_scale(raw) if scale.usable
                          else -np.inf)
        self._i += 1
        self._last_channel_values = values
        return max(values)

    def score_stream(self, episode: Episode) -> np.ndarray:
        self.start_episode()
        return np.array([self.score_step(episode.X[t]) for t in range(episode.T)])

    @property
    def last_winning_channel(self) -> str | None:
        """Which channel the most recent fusion took its value from."""
        if not self._last_channel_values:
            return None
        return self.channels[int(np.argmax(self._last_channel_values))]

    # ----------------------------------------------------------- internals

    def _raw_channel_streams(self, episode: Episode) -> list[np.ndarray]:
        """Per-channel causal stream, before normalization.

        Runs the real scoring path rather than a reconstruction of it, so the
        scales are learned from exactly the quantity that will be normalized.
        """
        for sub in self.subs:
            sub.start_episode()
        out = [np.zeros(episode.T) for _ in self.subs]
        for t in range(episode.T):
            scored_so_far = max(0, t - _WASHOUT + 1)
            for index, sub in enumerate(self.subs):
                cusum = sub.score_step(episode.X[t])
                if self.statistic == "cusum":
                    out[index][t] = cusum
                elif self.statistic == "rate":
                    out[index][t] = cusum / max(scored_so_far, 1)
                else:
                    out[index][t] = sub.last_raw_increment
        return out


def pick_fused_threshold(monitor: NormalizedFusionMonitor,
                         healthy_episodes: list[Episode], *,
                         fa_budget: float = 0.05) -> tuple[float, int, int]:
    """Threshold for the FUSED statistic, from healthy episodes only.

    Returns (theta, n_used, n_skipped_unscoreable). The skipped count is
    returned rather than swallowed: an episode the detector cannot score is
    not a quiet healthy episode, and silently folding it in as a zero is how
    a threshold ends up set by a distribution that is mostly absence of
    measurement.
    """
    streams, skipped = [], 0
    for ep in healthy_episodes:
        if not scoreable(ep):
            skipped += 1
            continue
        streams.append(monitor.score_stream(ep))
    if not streams:
        raise ValueError("no scoreable healthy episode to threshold on")
    theta = float(pick_threshold(streams, fa_budget=fa_budget,
                                 warn_infeasible=False))
    if not np.isfinite(theta):
        raise ValueError(f"degenerate fused threshold ({theta!r})")
    return theta, len(streams), skipped


if __name__ == "__main__":
    from derail.telemetry.adapter import episode_from_trace

    def _steps(seed: int, n: int = 10) -> list[dict]:
        rng = np.random.default_rng(seed)
        return [{"text": f"step {t} {rng.integers(0, 1000)}",
                 "token_logprobs": [-0.1] * 10, "action": "tool_call",
                 "latency_s": float(rng.uniform(0.3, 0.8)), "output_tokens": 10,
                 "error": False} for t in range(n)]

    healthy = [episode_from_trace(_steps(i), f"h{i}",
                                  use_sentence_transformers=False, extended=True)
               for i in range(60)]
    monitor = NormalizedFusionMonitor(Standardizer().fit(healthy[:40]),
                                      channels=("e", "m"), statistic="rate")
    monitor.fit(healthy[:40])
    monitor.fit_channel_scales(healthy[40:50])
    theta, used, skipped = pick_fused_threshold(monitor, healthy[50:])
    assert used == 10 and skipped == 0, (used, skipped)
    assert np.isfinite(theta)
    stream = monitor.score_stream(healthy[50])
    assert stream.shape == (healthy[50].T,)
    assert monitor.last_winning_channel in ("e", "m")
    short = episode_from_trace(_steps(999, n=2), "short",
                               use_sentence_transformers=False, extended=True)
    assert not scoreable(short)
    print("PASS normalized_fusion.py smoke test")
