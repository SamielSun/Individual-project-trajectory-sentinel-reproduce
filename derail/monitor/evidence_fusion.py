"""Fuse branches on calibrated tail evidence, with an operator that cannot dilute.

Averaging branches trades signal for noise. A spike of size S in one of k
branches becomes S/k, while the healthy null of the average shrinks only as
sqrt(k), so every branch that is inert on a given corpus costs the informative
ones about sqrt(k) of sensitivity. That is why no single averaged arm wins on
every corpus: the reliability channel earns its keep where tools fail and
subtracts where they do not, and the goal channel does the reverse.

Three parts, and they only work together:

  evidence  each branch's causal statistic is replaced by its upper-tail rank
            in that branch's OWN healthy null, in -log10 units. Branches then
            compare over the whole tail instead of at one reference quantile,
            and -- the property the union needs -- a branch sitting at its
            healthy floor scores exactly 0 rather than something negative.
  union     branches combine under a monotone rule. Evidence is non-negative,
            so a branch pinned at its healthy floor adds 0 under every rule
            here and leaves the combined statistic pointwise unchanged: its
            threshold and its detections are unchanged too, which is the
            invariant averaging violates.
  omnibus   which rule is right depends on the failure, not the corpus, so
            all three are computed, each is calibrated against its own healthy
            null exactly as a branch is, and the maximum is taken. Selecting
            one rule per corpus would be selecting by benchmark identity.

Thresholding is unchanged in kind: one threshold on the combined statistic,
taken on a calibration fold disjoint from both the fit and the null. Nothing
here sees a label, an onset, a class, a framework or a corpus name; the null
is healthy development data and the combination is fixed in advance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Branch combination rules. `max` is Tippett's: the union fires on the single
#: strongest branch, and is the only one of the three that cannot be diluted.
#: `sum` is Fisher's and aggregates evidence spread thinly across branches, at
#: the cost of accumulating the noise of the inert ones. `top2` is between.
#:
#: Neither wins everywhere, and which one wins is a property of the failure
#: rather than of the corpus: a tool cascade lights one branch hard, so max
#: reads it and sum averages it away; goal drift lifts several branches a
#: little, so sum accumulates it and max sees nothing decisive. Choosing per
#: corpus would be choosing by benchmark identity, which is not a detector.
#:
#: `omnibus` is the way out. The three rules are computed, each is calibrated
#: against ITS OWN healthy null exactly as a branch is, and the maximum of
#: those calibrated values is taken. Because the second level is on the same
#: evidence scale, it inherits the same non-dilution property, and the cost is
#: a multiplicity of three rather than a choice between them.
RULES = ("max", "sum", "top2")
COMBINERS = RULES + ("omnibus",)

#: Values within this relative tolerance of a null value count as ties. The
#: healthy null of a quiet branch is constant only to within floating point,
#: and without a tolerance a 1e-15 wobble would read as beyond everything seen.
_TIE_TOL = 1e-9

#: Above the largest healthy value the empirical rank saturates and every
#: departure ties at the ceiling, which is fatal at exactly the place the
#: threshold sits: on a 16-episode scale fold of ~7 scoreable steps the whole
#: null is ~100 values, so healthy episodes reach the ceiling routinely and a
#: rank rule cannot split an atom -- they all alarm together. The tail is
#: therefore continued past the ceiling, using the mean excess of the null's
#: own upper tail as its unit.
#:
#: The continuation is logarithmic in that unit, not linear. A linear
#: continuation with a hard cap was tried first and only moved the atom from
#: the ceiling to the cap: the top healthy episodes piled up there instead,
#: and the false-alarm rate stayed near 6-10%. Growing as log10(1 + excess/
#: beta) keeps ranking however far out a value lands, which is also the
#: conservative reading -- beyond everything the fold saw, the tail shape is
#: unknown, and a heavy-tailed continuation refuses to treat a large excess
#: as overwhelming evidence.
#:
#: Extrapolation still stops somewhere: at most as far beyond the fold's
#: resolution as the fold itself resolves, so evidence stays within twice the
#: ceiling and no branch can be handed an unbounded score by a small beta.
#: Exceedances are counted from the 80th percentile rather than the 95th: at
#: n ~ 100 the higher cut leaves 5 or 6 of them, too few to average.
_POT_QUANTILE = 0.80
_MIN_EXCESS = 5


def _tail_scale(arr: np.ndarray) -> float:
    """Mean excess of the healthy upper tail, with one fallback.

    The fallback fits the same exponential through two points the null always
    has -- its median and its largest value -- and is what keeps a short fold
    from collapsing to a flat ceiling. A null with no spread at all has no
    tail to fit and returns 0, which is reported rather than invented.
    """
    cut = float(np.quantile(arr, _POT_QUANTILE))
    excess = arr[arr > cut] - cut
    if excess.size >= _MIN_EXCESS and excess.mean() > 0:
        return float(excess.mean())
    span = float(arr[-1] - np.median(arr))
    return (span / float(np.log(max(arr.size, 4) / 2.0))) if span > 0 else 0.0


@dataclass(frozen=True)
class TailEvidence:
    """One branch's healthy null as an upper-tail rank, in -log10 units.

    `resolution_limited` marks a null with no estimable tail: its evidence
    saturates at `ceiling` and cannot rank departures against each other. That
    is the honest reading of a branch whose healthy stream never varies, and
    it is recorded rather than papered over with an invented scale.
    """

    null: np.ndarray            # ascending, finite
    tol: float
    beta: float                 # unit of the upper tail; 0 if not estimable

    @property
    def n(self) -> int:
        return int(self.null.size)

    @property
    def ceiling(self) -> float:
        return float(np.log10(1.0 + self.n))

    @property
    def resolution_limited(self) -> bool:
        return self.beta <= 0.0

    @classmethod
    def fit(cls, values) -> "TailEvidence":
        arr = np.asarray(list(values), dtype=float).ravel()
        arr = np.sort(arr[np.isfinite(arr)])
        if arr.size == 0:
            raise ValueError("no finite healthy value to calibrate evidence on")
        span = float(arr[-1] - arr[0])
        tol = _TIE_TOL * max(span, abs(float(arr[-1])), 1.0)
        return cls(null=arr, tol=tol, beta=_tail_scale(arr))

    def evidence(self, values) -> np.ndarray:
        """-log10 of the empirical upper-tail probability. Non-negative."""
        v = np.asarray(values, dtype=float)
        finite = np.isfinite(v)
        # count = how many healthy values are at least this large.
        count = self.n - np.searchsorted(self.null, np.where(finite, v, 0.0)
                                         - self.tol, side="left")
        out = -np.log10((count + 1.0) / (self.n + 1.0))
        beyond = count == 0
        if self.beta > 0 and np.any(beyond):
            excess = np.maximum(v - self.null[-1], 0.0)
            ext = np.minimum(np.log10(1.0 + excess / self.beta), self.ceiling)
            out = np.where(beyond, self.ceiling + ext, out)
        # A step the detector could not score is an absent measurement, not
        # quiet evidence, and must not be able to satisfy a threshold.
        return np.where(finite, out, -np.inf)


class EvidenceUnion:
    """Branch-wise calibrated evidence under a monotone combination rule."""

    def __init__(self, names: tuple[str, ...], combiner: str = "max") -> None:
        if combiner not in COMBINERS:
            raise ValueError(f"combiner must be one of {COMBINERS}, "
                             f"got {combiner!r}")
        if not names:
            raise ValueError("a union needs at least one branch")
        self.names = tuple(names)
        self.combiner = combiner
        self.calibrators: dict[str, TailEvidence] = {}
        #: omnibus only: one per rule, over that rule's own healthy null.
        self.rule_calibrators: dict[str, TailEvidence] = {}

    def fit(self, null_streams: dict[str, list]) -> "EvidenceUnion":
        """Learn each branch's healthy null. Healthy scale-fold streams only."""
        for name in self.names:
            self.calibrators[name] = TailEvidence.fit(
                np.concatenate([np.asarray(s, dtype=float).ravel()
                                for s in null_streams[name]]))
        if self.combiner == "omnibus":
            # The second level scores whole steps, so the branches have to be
            # the same steps of the same episodes -- not merely the same
            # number of values.
            shapes = {n: [np.shape(s) for s in null_streams[n]]
                      for n in self.names}
            if len({tuple(v) for v in shapes.values()}) != 1:
                raise ValueError(
                    "omnibus calibrates a second level over the branches "
                    "jointly, so every branch must supply the same episodes "
                    f"step for step; got {shapes}")
            n_eps = len(next(iter(null_streams.values())))
            per_episode = [{n: np.asarray(null_streams[n][i], dtype=float)
                            for n in self.names} for i in range(n_eps)]
            for rule in RULES:
                self.rule_calibrators[rule] = TailEvidence.fit(
                    np.concatenate([self._apply(rule, self.evidence(s))
                                    for s in per_episode]))
        return self

    def evidence(self, streams: dict[str, np.ndarray]) -> np.ndarray:
        """Per-branch evidence for one episode, shaped (T, n_branches)."""
        if not self.calibrators:
            raise RuntimeError("fit() must run before evidence()")
        return np.stack([self.calibrators[n].evidence(streams[n])
                         for n in self.names], axis=1)

    @staticmethod
    def _apply(rule: str, ev: np.ndarray) -> np.ndarray:
        if rule == "max":
            return ev.max(axis=1)
        # Absent measurements must not poison an aggregate the way they cannot
        # poison a max, so they contribute nothing and leave the step absent
        # only if every branch is absent.
        absent = ~np.isfinite(ev)
        filled = np.where(absent, 0.0, ev)
        total = (filled.sum(axis=1) if rule == "sum"
                 else np.sort(filled, axis=1)[:, -2:].sum(axis=1))
        return np.where(absent.all(axis=1), -np.inf, total)

    def combine(self, streams: dict[str, np.ndarray]) -> np.ndarray:
        """The combined causal statistic for one episode."""
        ev = self.evidence(streams)
        if self.combiner != "omnibus":
            return self._apply(self.combiner, ev)
        return np.stack([self.rule_calibrators[r].evidence(self._apply(r, ev))
                         for r in RULES], axis=1).max(axis=1)

    @property
    def resolution_limited(self) -> tuple[str, ...]:
        return tuple(n for n in self.names
                     if self.calibrators[n].resolution_limited)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    null = rng.normal(size=4000)
    cal = TailEvidence.fit(null)
    assert cal.evidence(null.min() - 1.0) == 0.0
    assert cal.evidence(0.0) > 0.0
    assert cal.evidence(1e9) <= 2 * cal.ceiling + 1e-9
    grid = np.linspace(-5, 10, 400)
    assert np.all(np.diff(cal.evidence(grid)) >= -1e-12), "evidence must be monotone"

    quiet = np.zeros(4000)
    flat = TailEvidence.fit(quiet)
    assert flat.resolution_limited and flat.evidence(0.0) == 0.0
    assert flat.evidence(1e-15) == 0.0, "a float wobble is not a departure"
    assert abs(float(flat.evidence(1.0)) - flat.ceiling) < 1e-12

    union = EvidenceUnion(("a", "b"), "max").fit(
        {"a": [null], "b": [quiet]})
    alone = EvidenceUnion(("a",), "max").fit({"a": [null]})
    probe = rng.normal(size=50)
    both = union.combine({"a": probe, "b": np.zeros(50)})
    one = alone.combine({"a": probe})
    assert np.array_equal(both, one), "an inert branch must not change the union"
    print("PASS evidence_fusion.py smoke test")
