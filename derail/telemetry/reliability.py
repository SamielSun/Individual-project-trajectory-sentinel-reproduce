"""Tool-reliability channel r_t: the SHAPE of tool failure over an episode.

`error_flag` is per-step and memoryless, so an isolated flake and the third
step of a cascade look the same. Healthy runs already carry errors on about 9%
of steps, so error presence has almost no contrast; what separates a cascade is
its scope and its failure to clear -- how many distinct tools are affected,
whether the same call fails again, how long since anything worked.

Every dim is causal (steps 0..t only), bounded in [0, 1], oriented so HIGHER =
more anomalous, and derived only from tool names, argument keys and error flags.
No labels, no task or framework identity, no lookahead. Counts are squashed
through 1 - exp(-n/3) rather than used raw: an integer count has a lumpy healthy
null that a small calibration fold estimates badly.

A trace with no tool events leaves all dims 0 -- inert, never an error.
"""
from __future__ import annotations

import collections
import math

import numpy as np

from derail.common import (D_REL, D_TOTAL_GOAL, IDX_REL_BURST,
                           IDX_REL_CONSEC_FAIL, IDX_REL_DISTINCT_FAILING,
                           IDX_REL_FAIL_FRAC, IDX_REL_NO_RECOVERY,
                           IDX_REL_REPEAT_FAIL, IDX_REL_SINCE_CLEAN, REL_SLICE)
from derail.telemetry.events import parse_step_events

_SQUASH = 3.0
_BURST_WINDOW = 3
#: The IDX_REL_* names are absolute indices into the full feature vector; this
#: block returns only its own 7 dims, so they are shifted down by the base.
_B = D_TOTAL_GOAL


def _squash(n: float) -> float:
    """Bounded, continuous stand-in for a count."""
    return 1.0 - math.exp(-max(0.0, float(n)) / _SQUASH)


class ReliabilityState:
    """Causal per-episode tool-reliability state."""

    def __init__(self) -> None:
        self.seen_tool = False
        self.failing: set[str] = set()          # failed, not yet recovered
        self.tools_seen: set[str] = set()
        self.ever_failed: set[str] = set()
        self.recovered: set[str] = set()
        self.failed_calls: set[tuple[str, str]] = set()
        self.consec_fail = 0                    # consecutive failing CALLS
        self.steps_since_clean = 0
        self.recent = collections.deque(maxlen=_BURST_WINDOW)


def reliability_features(step: dict, state: ReliabilityState) -> np.ndarray:
    """One step dict -> the 7-dim r_t. Mutates `state`."""
    x = np.zeros(D_REL)
    events, _reasoning = parse_step_events(step)

    if not events:
        # No call this step. Time since the last clean result still runs, but
        # only once the episode has actually used a tool -- otherwise a purely
        # reasoning prefix would read as degradation.
        if state.seen_tool:
            state.steps_since_clean += 1
            state.recent.append(0.0)
            x[IDX_REL_SINCE_CLEAN - _B] = _squash(state.steps_since_clean)
            x[IDX_REL_DISTINCT_FAILING - _B] = (
                len(state.failing) / max(1, len(state.tools_seen)))
            x[IDX_REL_NO_RECOVERY - _B] = (
                len(state.ever_failed - state.recovered)
                / max(1, len(state.ever_failed)))
            x[IDX_REL_CONSEC_FAIL - _B] = _squash(state.consec_fail)
            x[IDX_REL_BURST - _B] = float(np.mean(state.recent)) if state.recent else 0.0
        return x

    state.seen_tool = True
    n_fail = n_repeat = 0
    clean_this_step = False
    for ev in events:
        state.tools_seen.add(ev.name)
        key = (ev.name, ev.args_key)
        if ev.is_error:
            n_fail += 1
            state.consec_fail += 1
            state.failing.add(ev.name)
            state.ever_failed.add(ev.name)
            if key in state.failed_calls:
                n_repeat += 1
            state.failed_calls.add(key)
        else:
            state.consec_fail = 0
            clean_this_step = True
            if ev.name in state.failing:
                state.failing.discard(ev.name)
                state.recovered.add(ev.name)

    fail_frac = n_fail / len(events)
    state.recent.append(fail_frac)
    state.steps_since_clean = 0 if clean_this_step else state.steps_since_clean + 1

    x[IDX_REL_FAIL_FRAC - _B] = fail_frac
    x[IDX_REL_DISTINCT_FAILING - _B] = (
        len(state.failing) / max(1, len(state.tools_seen)))
    x[IDX_REL_REPEAT_FAIL - _B] = n_repeat / n_fail if n_fail else 0.0
    x[IDX_REL_CONSEC_FAIL - _B] = _squash(state.consec_fail)
    x[IDX_REL_SINCE_CLEAN - _B] = _squash(state.steps_since_clean)
    x[IDX_REL_NO_RECOVERY - _B] = (len(state.ever_failed - state.recovered)
                              / max(1, len(state.ever_failed)))
    x[IDX_REL_BURST - _B] = float(np.mean(state.recent))
    return x


def _selftest() -> None:
    def s(names_errors):
        return {"text": "", "tool_events": [
            {"name": n, "args": {"q": a}, "result": "x", "is_error": e}
            for n, a, e in names_errors]}

    st = ReliabilityState()
    clean = reliability_features(s([("search", "a", False)]), st)
    assert clean.sum() == 0.0, "a clean call is not anomalous"

    st = ReliabilityState()
    reliability_features(s([("search", "a", True)]), st)
    second = reliability_features(s([("search", "a", True)]), st)
    assert second[IDX_REL_REPEAT_FAIL - _B] == 1.0, "same call failing twice"
    assert second[IDX_REL_CONSEC_FAIL - _B] > _squash(1), "consecutive failures grow"
    assert second[IDX_REL_NO_RECOVERY - _B] == 1.0

    st = ReliabilityState()
    reliability_features(s([("a", "q", True), ("b", "q", True)]), st)
    spread = reliability_features(s([("c", "q", True)]), st)
    assert spread[IDX_REL_DISTINCT_FAILING - _B] == 1.0, "all three tools failing"

    st = ReliabilityState()
    reliability_features(s([("a", "q", True)]), st)
    back = reliability_features(s([("a", "q2", False)]), st)
    assert back[IDX_REL_DISTINCT_FAILING - _B] == 0.0, "recovery clears the tool"
    assert back[IDX_REL_NO_RECOVERY - _B] == 0.0
    assert back[IDX_REL_SINCE_CLEAN - _B] == 0.0

    st = ReliabilityState()
    assert reliability_features({"text": "just thinking"}, st).sum() == 0.0, (
        "a trace with no tool events must stay inert")

    assert REL_SLICE.stop - REL_SLICE.start == D_REL
    print("reliability selftest ok")


if __name__ == "__main__":
    _selftest()
