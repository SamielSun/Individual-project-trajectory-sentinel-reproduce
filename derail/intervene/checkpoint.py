"""Where to resume from, decided only from what is knowable at alarm time.

The shipped rule resumes just after the last step whose tool call did not
return an error. "Did not error" is the wrong question. A `wrong_document`
result and a `context_corruption` result both arrive with `is_error=False`
and perfectly ordinary shape -- measured on the rollback corpus, the run that
searched arXiv five times and got the 1994 World Cup final back every time
records five clean successes. Resuming after them keeps the poisoned evidence
in the rebuilt history and spends the retry budget that was needed to redo it.

So the checkpoint asks whether a step's evidence is GROUNDED, not whether the
call returned. Groundedness comes from the content-grounding channel (g,
telemetry v4): result against its own query, against the agent's reasoning,
against the run's other results, plus character and JSON shape. A reference
fitted on the deployment's own healthy runs sets the level above which a step
is not trusted.

This is a strict refinement of the shipped rule, not a replacement: it moves
the checkpoint earlier exactly when the content channel says the evidence is
untrustworthy, and returns the same answer otherwise. Measured on the rollback
corpus it fires on 5 of 16 injected episodes -- every `multi_city_weather`
one, none of the arXiv ones, whose corrupted documents sit inside that task's
own healthy grounding null. That is a real limit of the channel on long
free-text results and it is reported rather than tuned away.

What this rule may see, and the signature is the enforcement: the steps so far
and a reference fitted before serving. What it may not see, and cannot reach
through any parameter here: the injected onset, the failure class, the final
verdict, the task's `success_fn`, or any tool result from a step after the one
being judged. `success_fn` exists to score the repaired run afterwards and is
never consulted while repairing.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from derail.common import Episode
from derail.monitor.grounding import GRD_DIM_NAMES, GroundingMonitor
from derail.telemetry.events import parse_step_events

#: Bumped when the rule changes shape, so a recorded checkpoint says which
#: rule produced it and an old result is never read as a new one.
CHECKPOINT_RULE_VERSION = "last_grounded_successful/v1"

#: Quantile of the healthy per-step grounding score above which a step is not
#: trusted. A quantile rather than the healthy MAXIMUM, which is where the
#: `_g_trip` precedent sits: a maximum over a few thousand steps is an extreme
#: order statistic set by one episode, and measured on the detector's own fit
#: corpora it came out at 25.4 -- above every step in the rollback corpus,
#: injected or not, so the rule could never fire.
#:
#: The errors here are not symmetric, which is what allows a quantile this
#: tight. Calling a good step ungrounded resumes slightly earlier and costs
#: some retained evidence; calling a poisoned step grounded keeps the poison
#: in the rebuilt history and spends the retry budget on it. So the cut is set
#: where healthy traffic is essentially never touched but real corruption
#: still trips it: measured, 0.95 leaves 0 of 21 held-out healthy steps
#: ungrounded and flags 5 of 16 injections, and loosening to 0.90 buys one
#: more injection for 9.5% of healthy steps.
TRIP_QUANTILE = 0.95


@dataclass(frozen=True)
class GroundingReference:
    """Healthy location and scale for the grounding dims, plus the trip level.

    Frozen before serving and carried as data, so the checkpoint rule at
    serving time cannot be influenced by the episode it is judging.
    """

    dims: tuple[str, ...]
    median: tuple[float, ...]
    scale: tuple[float, ...]
    #: Level above which a step's evidence is not TRUSTED, for choosing where
    #: to resume. A quantile: the cost of being wrong is asymmetric (see
    #: TRIP_QUANTILE), so it is set tight.
    trip: float
    n_healthy_episodes: int
    n_healthy_steps: int
    #: Level above which the channel may RAISE AN ALARM by itself, and a
    #: different question from `trip`. Alarming wrongly costs a healthy
    #: intervention, so this is the level no healthy step reached at all --
    #: the `_g_trip` rule from DESIGN.md amendment 4. Because nothing healthy
    #: reaches it, an override here adds no false alarms and therefore does
    #: not compete for the fused budget the behavioural channels share, which
    #: is the whole reason it can be added without costing the other classes.
    alarm_trip: float = float("inf")
    rule_version: str = CHECKPOINT_RULE_VERSION

    def score(self, x_t: np.ndarray) -> float:
        """One-sided robust z, maxed over dims -- the grounding monitor's own
        statistic, recomputed from the stored reference rather than from a
        live fit."""
        z = (np.asarray(x_t, dtype=float)[list(_dim_indices(self.dims))]
             - np.asarray(self.median)) / np.asarray(self.scale)
        return float(np.max(np.clip(z, 0.0, GroundingMonitor._Z_CAP),
                            initial=0.0))

    def is_grounded(self, x_t: np.ndarray) -> bool:
        return self.score(x_t) <= self.trip

    def should_alarm(self, x_t: np.ndarray) -> bool:
        """Is this step outside anything healthy ever produced?"""
        return self.score(x_t) > self.alarm_trip

    def as_record(self) -> dict:
        record = asdict(self)
        for key in ("dims", "median", "scale"):
            record[key] = list(record[key])
        return record

    @classmethod
    def from_record(cls, record: dict) -> "GroundingReference":
        return cls(dims=tuple(record["dims"]),
                   median=tuple(record["median"]),
                   scale=tuple(record["scale"]),
                   trip=float(record["trip"]),
                   n_healthy_episodes=int(record["n_healthy_episodes"]),
                   n_healthy_steps=int(record["n_healthy_steps"]),
                   alarm_trip=float(record.get("alarm_trip", float("inf"))),
                   rule_version=record.get("rule_version",
                                           CHECKPOINT_RULE_VERSION))


def _dim_indices(dims: tuple[str, ...]) -> tuple[int, ...]:
    from derail.common import GRD_SLICE
    return tuple(GRD_SLICE.start + GRD_DIM_NAMES.index(d) for d in dims)


def fit_grounding_reference(healthy_episodes: list[Episode],
                            dims: tuple[str, ...] = GRD_DIM_NAMES
                            ) -> GroundingReference:
    """Fit the reference on healthy runs only. No label reaches this.

    Uses `GroundingMonitor` to do the fitting rather than repeating its robust
    statistics here, so the checkpoint rule and the grounding channel can
    never disagree about what a healthy step looks like.

    Fit on the DEPLOYMENT's own healthy traffic, the same principle the
    detector threshold already follows. What a grounded result looks like is a
    property of the tools and the task: a reference fitted on a long research
    survey put ordinary arXiv results 15 standard units from its own centre,
    because it had never seen one.
    """
    if not healthy_episodes:
        raise ValueError("no healthy episodes to fit a grounding reference on")
    monitor = GroundingMonitor(dims=dims)
    monitor.fit(healthy_episodes)
    scores = [monitor.score_step(ep.X[t])
              for ep in healthy_episodes for t in range(ep.T)]
    return GroundingReference(
        dims=tuple(dims),
        median=tuple(float(v) for v in monitor._med),
        scale=tuple(float(v) for v in monitor._scale),
        trip=float(np.quantile(scores, TRIP_QUANTILE)),
        n_healthy_episodes=len(healthy_episodes),
        n_healthy_steps=len(scores),
        alarm_trip=float(max(scores)))


def step_is_grounded(step: dict, x_t: np.ndarray,
                     reference: GroundingReference) -> bool:
    """Did this step produce evidence worth keeping?

    Two conditions, both causal: the call has to have returned without error,
    and its CONTENT has to sit inside what healthy runs produced. A step with
    no tool call carries no evidence, so it is not a place to resume from --
    it is neither grounded nor ungrounded, and callers treat it as carrying
    nothing.
    """
    events, _ = parse_step_events(step)
    if not events or all(e.is_error for e in events):
        return False
    return reference.is_grounded(x_t)


def last_grounded_step_before(steps: list[dict], X: np.ndarray, limit: int,
                              reference: GroundingReference) -> int:
    """Resume index: just after the last step that produced trusted evidence.

    0 when no such step exists, which is a real answer -- the run had nothing
    worth keeping, so the retry starts from the task.
    """
    resume = 0
    for t, step in enumerate(steps[:limit]):
        if step_is_grounded(step, X[t], reference):
            resume = t + 1
    return resume


def discarded_evidence_calls(steps: list[dict], start: int
                             ) -> list[tuple[str, dict]]:
    """Distinct successful tool calls the rollback is about to throw away.

    Truncating at the checkpoint discards everything the run did later,
    including lookups that were never implicated in the fault. On the rollback
    corpus that is a measured failure: the task requires a Wikipedia lookup,
    the lookup happened after the corrupted arXiv call, and rolling back threw
    it away -- the retry then had to spend budget rediscovering a fact the run
    already had, and on one episode did not.

    Returned so the caller can RE-ISSUE them, not replay them. Regeneration
    beats preservation here and the corpus shows why: the discarded span of
    `arxiv_paper_search-wrong_document-00` holds six grounded-looking steps
    whose results are about the 1994 World Cup, so replaying those
    observations would carry the corruption straight back into the rebuilt
    history. Re-issuing the same call gets a fresh result instead.

    Errored calls are skipped -- they produced nothing to restore -- and
    repeats are collapsed, so a run that made the same call eight times costs
    one regeneration. Order is preserved so the rebuilt evidence arrives the
    way the run gathered it. Whether a call is SAFE to re-issue is the effect
    policy's decision, not this function's.
    """
    seen, out = set(), []
    for step in steps[start:]:
        for event in parse_step_events(step)[0]:
            if event.is_error:
                continue
            key = (event.name, event.args_key)
            if key in seen:
                continue
            seen.add(key)
            out.append((event.name, dict(event.args or {})))
    return out


def write_reference(reference: GroundingReference, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference.as_record(), indent=2) + "\n", "utf-8")
    return path


def read_reference(path: Path) -> GroundingReference:
    return GroundingReference.from_record(
        json.loads(path.read_text("utf-8")))
