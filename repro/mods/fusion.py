"""Modification M1 — replace the per-channel **max** fusion with a **mean** fusion.

Why this modification
---------------------
The upstream README states the mechanism claim we are testing:

    "the per-channel max-fusion wrapper, not the reservoir, carries most of the
     margin"

i.e. the gain of the primary monitor ``esn_cusum_max`` over the monolithic ESN
is attributed to how the three per-channel CUSUM streams are *combined*, not to
the reservoir that produces them. ``esn_cusum_max`` combines them with ``max``
(``derail/monitor/esn.py``, ``ChannelMaxESNMonitor.score_step``).

We change exactly one thing: the combination rule. Everything else — the same
three sub-monitors, the same reservoir weights and seeds, the same CUSUM, the
same thresholds picked on the healthy validation split at the same 5% budget —
is held fixed. If the claim is right, replacing ``max`` with ``mean`` must cost
a large part of the margin; if the margin survives, the claim is misattributed.

Isolation
---------
Only the monitor literally named ``esn_cusum_max`` is modified. The other
channel variants (``esn_cusum[e]``, ``esn_cusum[e,u]``, ...) keep ``max`` and
stay in the table as an untouched control row set.

Fusions implemented
-------------------
``mean``     arithmetic mean of the three CUSUM streams (dilutes a single-channel
             excursion, which is exactly the failure mode the paper attributes
             the gain to)
``top2mean`` mean of the two largest streams — a graded control that keeps most
             of max's peak sensitivity while recovering some smoothing
"""

from __future__ import annotations

import numpy as np

TARGET_MONITOR = "esn_cusum_max"

AGGREGATORS = {
    "mean": lambda vals: float(np.mean(vals)),
    "top2mean": lambda vals: float(np.mean(sorted(vals)[-2:])),
}


def _aggregate_array(rows: np.ndarray, fusion: str) -> np.ndarray:
    if fusion == "mean":
        return np.mean(rows, axis=0)
    if fusion == "top2mean":
        return np.mean(np.sort(rows, axis=0)[-2:], axis=0)
    raise ValueError(f"unknown fusion {fusion!r}")


def apply(fusion: str = "mean") -> None:
    """Patch ``ChannelMaxESNMonitor`` in place for the current process."""
    if fusion not in AGGREGATORS:
        raise ValueError(f"unknown fusion {fusion!r}; choose from {sorted(AGGREGATORS)}")

    from derail.monitor.esn import ChannelMaxESNMonitor

    if getattr(ChannelMaxESNMonitor, "_repro_fusion_patch", None) == fusion:
        return

    agg = AGGREGATORS[fusion]
    orig_score_step = ChannelMaxESNMonitor.score_step
    orig_components = ChannelMaxESNMonitor.score_episode_components

    def score_step(self, x_t: np.ndarray) -> float:
        if self.name != TARGET_MONITOR:
            return orig_score_step(self, x_t)
        return agg([sub.score_step(x_t) for sub in self.subs])

    def score_episode_components(self, episode):
        if self.name != TARGET_MONITOR:
            return orig_components(self, episode)
        parts = [sub.score_episode_components(episode) for sub in self.subs]
        keys = ("fused", "surprise", "disagreement")
        return {
            key: _aggregate_array(np.array([p[key] for p in parts]), fusion)
            for key in keys
        }

    ChannelMaxESNMonitor.score_step = score_step
    ChannelMaxESNMonitor.score_episode_components = score_episode_components
    ChannelMaxESNMonitor._repro_fusion_patch = fusion
